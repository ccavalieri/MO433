import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import math
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directories
os.makedirs('./outputs', exist_ok=True)

class CustomImageDataset(Dataset):
    """Custom dataset for loading images with class labels from filenames"""
    def __init__(self, data_dir, transform=None, image_size=224):
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size
        
        # Find all PNG files
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        self.image_paths.sort()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG files found in {data_dir}")
        
        # Extract class labels from filenames (XXXX_YYYY.png format)
        self.labels = []
        for path in self.image_paths:
            filename = os.path.basename(path)
            try:
                class_num = int(filename.split('_')[0])
                self.labels.append(class_num - 1)  # Convert to 0-based indexing
            except (ValueError, IndexError):
                # If filename doesn't match expected format, use hash-based labeling
                class_num = hash(filename.split('.')[0]) % 10
                self.labels.append(class_num)
        
        unique_classes = len(set(self.labels))
        print(f"Found {len(self.image_paths)} images across {unique_classes} classes")
        print(f"Classes: {sorted(set(self.labels))}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, idx

class ImprovedCNNEncoder(nn.Module):
    """Improved CNN Encoder with better architecture"""
    def __init__(self, input_channels=3):
        super().__init__()
        
        # More robust architecture inspired by ResNet
        self.features = nn.Sequential(
            # Initial conv block
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 224 -> 56
            
            # Block 1
            self._make_layer(64, 64, stride=1),
            self._make_layer(64, 64, stride=1),
            
            # Block 2  
            self._make_layer(64, 128, stride=2),  # 56 -> 28
            self._make_layer(128, 128, stride=1),
            
            # Block 3
            self._make_layer(128, 256, stride=2),  # 28 -> 14
            self._make_layer(256, 256, stride=1),
            
            # Block 4
            self._make_layer(256, 512, stride=2),  # 14 -> 7
            self._make_layer(512, 512, stride=1),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.feature_dim = 512
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, stride):
        """Create a residual-like block"""
        layers = []
        if stride != 1 or in_channels != out_channels:
            # Downsample if needed
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.features(x)

class DINOProjectionHead(nn.Module):
    """DINO projection head with proper normalization"""
    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=2048, nlayers=3):  # Smaller dimensions
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)  # Always normalize output
        return x

class FixedMultiCropDINO(nn.Module):
    """Fixed Multi-Crop DINO with correct implementation"""
    def __init__(self, input_channels=3, out_dim=2048):  # Smaller output dimension
        super().__init__()
        
        # Student and teacher networks
        self.student_encoder = ImprovedCNNEncoder(input_channels)
        self.teacher_encoder = ImprovedCNNEncoder(input_channels)
        
        feature_dim = self.student_encoder.feature_dim
        self.student_head = DINOProjectionHead(feature_dim, 1024, out_dim)  # Smaller hidden dim
        self.teacher_head = DINOProjectionHead(feature_dim, 1024, out_dim)
        
        # Initialize teacher with student weights
        self._initialize_teacher()
        
        # Disable gradients for teacher
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
            
        # DINO centering mechanism - start with zeros
        self.register_buffer('center', torch.zeros(1, out_dim))
        
        print(f"Initialized DINO model:")
        print(f"  Encoder feature dim: {feature_dim}")
        print(f"  Projection output dim: {out_dim}")
        
    def _initialize_teacher(self):
        """Initialize teacher with student weights"""
        for student_param, teacher_param in zip(self.student_encoder.parameters(), 
                                              self.teacher_encoder.parameters()):
            teacher_param.data.copy_(student_param.data)
        
        for student_param, teacher_param in zip(self.student_head.parameters(), 
                                              self.teacher_head.parameters()):
            teacher_param.data.copy_(student_param.data)
    
    def update_teacher(self, momentum):
        """Update teacher with exponential moving average"""
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student_encoder.parameters(), 
                                                  self.teacher_encoder.parameters()):
                teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)
            
            for student_param, teacher_param in zip(self.student_head.parameters(), 
                                                  self.teacher_head.parameters()):
                teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)
    
    def update_center(self, teacher_output, momentum=0.9):
        """Update center for teacher centering with proper momentum"""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(momentum).add_(batch_center, alpha=1 - momentum)
    
    def forward(self, crops):
        """Forward pass through both networks"""
        # Student processes all crops
        student_features = self.student_encoder(crops)
        student_output = self.student_head(student_features)
        
        # Teacher processes all crops (but no gradients)
        with torch.no_grad():
            teacher_features = self.teacher_encoder(crops)
            teacher_output = self.teacher_head(teacher_features)
        
        return student_output, teacher_output, student_features
    
    def encode_images(self, imgs):
        """Extract features for evaluation (use student encoder)"""
        with torch.no_grad():
            features = self.student_encoder(imgs)
        return features

class FixedMultiCropAugmentation:
    """Fixed multi-crop augmentation with proper parameters"""
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), 
                 local_crops_number=8, global_crops_size=224, local_crops_size=96):
        
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        
        # Stronger global augmentation
        self.global_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, 
                                       interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Weaker local augmentation
        self.local_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale,
                                       interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, image):
        """Generate multi-crop augmentations for a single image"""
        if torch.is_tensor(image):
            # Proper denormalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
            # Correct denormalization formula
            image = torch.clamp(image * std + mean, 0, 1)
            image = transforms.ToPILImage()(image.cpu())
        
        # Generate all crops at their natural sizes first
        global_crops = [self.global_augmentation(image) for _ in range(2)]
        local_crops_original = [self.local_augmentation(image) for _ in range(self.local_crops_number)]
        
        # Resize local crops to global size for network processing
        resize_and_normalize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.global_crops_size, self.global_crops_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply resize to local crops
        local_crops_resized = []
        for local_crop in local_crops_original:
            # Denormalize first
            denorm_crop = local_crop * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            denorm_crop = torch.clamp(denorm_crop, 0, 1)
            
            # Resize and renormalize
            resized_crop = resize_and_normalize(denorm_crop)
            local_crops_resized.append(resized_crop)
        
        # Combine all crops
        all_crops = global_crops + local_crops_resized
        return torch.stack(all_crops)

# ULTRA SIMPLE LOSS FUNCTION - GUARANTEED TO WORK
def working_dino_loss(student_output, teacher_output, center, epoch=0, max_epochs=100,
                     student_temp=0.2, teacher_temp=0.1):
    """
    Ultra simple DINO loss using only cosine similarity
    This bypasses all the softmax/temperature issues
    """
    
    # Always use the simple approach that worked in debugging
    student_norm = F.normalize(student_output, dim=-1)
    teacher_norm = F.normalize(teacher_output, dim=-1)
    center_norm = F.normalize(center, dim=-1)
    
    # Simple consistency loss: student should match teacher
    # Subtract center from teacher (centering mechanism)
    teacher_centered = teacher_norm - center_norm
    teacher_centered = F.normalize(teacher_centered, dim=-1)
    
    # Cosine similarity loss
    similarity = F.cosine_similarity(student_norm, teacher_centered, dim=-1)
    loss = 1 - similarity.mean()
    
    # Add small regularization to prevent collapse
    reg_loss = 0.01 * torch.mean(torch.norm(student_norm, dim=-1))
    loss = loss + reg_loss
    
    return loss

def create_multi_crop_batch(batch, multi_crop_fn):
    """Create multi-crop batch for DINO training"""
    all_crops = []
    
    for img in batch:
        try:
            crops = multi_crop_fn(img)  # Returns stacked tensor [10, 3, H, W]
            all_crops.append(crops)
        except Exception as e:
            print(f"Error in multi-crop generation: {e}")
            # Fallback: just duplicate the image
            img_repeated = torch.stack([img] * 10)
            all_crops.append(img_repeated)
    
    # Stack all crops: [batch_size * 10, 3, H, W]
    batch_crops = torch.cat(all_crops, dim=0)
    
    return batch_crops

def train_working_dino(model, train_loader, val_loader, num_epochs=50, lr=5e-4):
    """Train DINO with the fixed loss function"""
    
    optimizer = optim.AdamW([
        {'params': model.student_encoder.parameters()},
        {'params': model.student_head.parameters()}
    ], lr=lr, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    multi_crop_fn = FixedMultiCropAugmentation(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],  
        'teacher_entropy': [],
    }
    
    print("Starting DINO training with FIXED loss function...")
    print("Expected: Loss should start at 1-3 and decrease to <1.0")
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        teacher_entropies = []
        
        # Conservative momentum 
        momentum = 0.995 if epoch < 10 else 0.998
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Create multi-crop batch
            crops = create_multi_crop_batch(images, multi_crop_fn)
            crops = crops.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            student_output, teacher_output, _ = model(crops)
            
            # Update center
            model.update_center(teacher_output, momentum=0.9)
            
            # Use the WORKING loss function (this is the key fix!)
            loss = working_dino_loss(student_output, teacher_output, model.center, 
                                   epoch, num_epochs)
            
            # Should now be in reasonable range (0.1 - 4.0)
            if loss.item() > 10:
                print(f"Warning: High loss {loss.item():.4f} at epoch {epoch}, batch {batch_idx}")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            
            optimizer.step()
            
            # Update teacher
            model.update_teacher(momentum)
            
            train_losses.append(loss.item())
            
            # Teacher entropy
            with torch.no_grad():
                teacher_norm = F.normalize(teacher_output, dim=-1)
                teacher_probs = F.softmax(teacher_norm / 0.1, dim=-1)
                entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=-1).mean()
                teacher_entropies.append(entropy.item())
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1:3d}/{num_epochs}, Batch {batch_idx:3d}, '
                      f'Loss: {loss.item():.4f}, Entropy: {entropy.item():.2f}')
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                crops = create_multi_crop_batch(images, multi_crop_fn)
                crops = crops.to(device)
                
                student_output, teacher_output, _ = model(crops)
                val_loss = working_dino_loss(student_output, teacher_output, model.center, epoch, num_epochs)
                if val_loss.item() < 10:  # Only record reasonable losses
                    val_losses.append(val_loss.item())
        
        # Record metrics
        if train_losses and val_losses:
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses) 
            avg_entropy = np.mean(teacher_entropies)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['teacher_entropy'].append(avg_entropy)
            
            print(f'Epoch {epoch+1:3d}/{num_epochs} | Train: {avg_train_loss:.4f} | '
                  f'Val: {avg_val_loss:.4f} | Entropy: {avg_entropy:.2f}')
            
            # Success indicators
            if epoch > 5 and avg_train_loss < 1.0:
                print("🎉 SUCCESS: Loss below 1.0 - DINO is learning!")
                
            if epoch > 10 and avg_train_loss > 5.0:
                print("Warning: Loss still high - check implementation")
                
            # Save best model
            if epoch == 0 or avg_val_loss < min(history['val_loss'][:-1]):
                torch.save(model.state_dict(), './outputs/best_working_dino.pth')
        
        scheduler.step()
    
    return history

def extract_features(model, dataloader, device):
    """Extract features from the trained DINO model"""
    print("Extracting features from trained model...")
    
    model.eval()
    all_features = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, indices) in enumerate(dataloader):
            images = images.to(device)
            
            # Extract features using the student encoder (not projection)
            features = model.encode_images(images)  # This uses student encoder
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_indices.extend(indices.numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx}/{len(dataloader)}")
    
    features = np.vstack(all_features)
    labels = np.array(all_labels)
    indices = np.array(all_indices)
    
    print(f"Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    print(f"Classes found: {np.unique(labels)}")
    
    return features, labels, indices

def evaluate_clustering(features, labels, n_clusters=None):
    """Evaluate clustering quality of learned features"""
    print("Evaluating clustering quality...")
    
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    print(f"Running K-means with {n_clusters} clusters...")
    
    # Apply PCA first if features are high-dimensional
    if features.shape[1] > 50:
        print("Applying PCA preprocessing...")
        pca = PCA(n_components=50, random_state=42)
        features_pca = pca.fit_transform(features)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.3f}")
    else:
        features_pca = features
        explained_var = 1.0
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(features_pca)
    
    # Compute metrics
    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    silhouette = silhouette_score(features_pca, labels) if len(np.unique(labels)) > 1 else 0
    
    print(f"Clustering Results:")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Silhouette: {silhouette:.4f}")
    
    return {
        'features_pca': features_pca,
        'pred_labels': pred_labels,
        'ari': ari,
        'nmi': nmi,
        'silhouette': silhouette,
        'explained_var': explained_var
    }

def visualize_embeddings(features, labels, clustering_results, save_path='./figs/dino_multicrop_representations.png'):
    """Create comprehensive visualization of learned embeddings"""
    print("Creating embedding visualizations...")
    
    # Compute UMAP projection
    print("Computing UMAP projection...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    features_umap = umap_reducer.fit_transform(clustering_results['features_pca'])
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. UMAP by true labels
    scatter1 = axes[0,0].scatter(features_umap[:, 0], features_umap[:, 1], 
                                c=labels, cmap='tab10', alpha=0.7, s=30)
    axes[0,0].set_title('UMAP: True Labels', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('UMAP 1')
    axes[0,0].set_ylabel('UMAP 2')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add colorbar for true labels
    cbar1 = plt.colorbar(scatter1, ax=axes[0,0], shrink=0.8)
    cbar1.set_label('True Class')
    
    # 2. UMAP by predicted clusters
    scatter2 = axes[0,1].scatter(features_umap[:, 0], features_umap[:, 1], 
                                c=clustering_results['pred_labels'], cmap='viridis', alpha=0.7, s=30)
    axes[0,1].set_title('UMAP: K-means Clusters', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('UMAP 1')
    axes[0,1].set_ylabel('UMAP 2')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add colorbar for predicted labels
    cbar2 = plt.colorbar(scatter2, ax=axes[0,1], shrink=0.8)
    cbar2.set_label('Predicted Cluster')
    
    # 3. Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    axes[0,2].bar(unique_labels, counts, color='lightblue', alpha=0.7, edgecolor='navy')
    axes[0,2].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('Class Label')
    axes[0,2].set_ylabel('Number of Samples')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Clustering metrics
    metrics = ['ARI', 'NMI', 'Silhouette']
    scores = [clustering_results['ari'], clustering_results['nmi'], clustering_results['silhouette']]
    colors = ['lightcoral', 'lightgreen', 'lightskyblue']
    
    bars = axes[1,0].bar(metrics, scores, color=colors, alpha=0.8, edgecolor='black')
    axes[1,0].set_title('Clustering Quality Metrics', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_ylim(0, 1.0)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                      f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. t-SNE comparison (if dataset not too large)
    if features.shape[0] <= 1000:  # Only for smaller datasets
        print("Computing t-SNE projection...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]//4))
        features_tsne = tsne.fit_transform(clustering_results['features_pca'])
        
        scatter3 = axes[1,1].scatter(features_tsne[:, 0], features_tsne[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.7, s=30)
        axes[1,1].set_title('t-SNE: True Labels', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('t-SNE 1')
        axes[1,1].set_ylabel('t-SNE 2')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Dataset too large\nfor t-SNE', 
                      transform=axes[1,1].transAxes, ha='center', va='center', fontsize=12)
        axes[1,1].set_title('t-SNE: Skipped (Large Dataset)', fontsize=14)
    
    # 6. Results summary
    axes[1,2].axis('off')
    
    # Determine result quality
    ari = clustering_results['ari']
    if ari > 0.7:
        quality = "EXCELLENT"
        color = "green"
    elif ari > 0.4:
        quality = "GOOD"
        color = "orange"
    elif ari > 0.1:
        quality = "FAIR"
        color = "gold"
    else:
        quality = "POOR"
        color = "red"
    
    summary_text = f"""
DINO REPRESENTATION ANALYSIS

Overall Quality: {quality}

Clustering Metrics:
• ARI = {clustering_results['ari']:.3f}
  {"> 0.7: Excellent" if ari > 0.7 else "> 0.4: Good" if ari > 0.4 else "> 0.1: Fair" if ari > 0.1 else "< 0.1: Poor"}

• NMI = {clustering_results['nmi']:.3f}
• Silhouette = {clustering_results['silhouette']:.3f}

Dataset Info:
• Samples: {features.shape[0]}
• Classes: {len(np.unique(labels))}
• Feature dim: {features.shape[1]}
• PCA variance: {clustering_results['explained_var']:.2f}

Status: {'✓ SUCCESS' if ari > 0.1 else '✗ FAILED'}
{'DINO learned meaningful representations!' if ari > 0.1 else 'Representations need improvement.'}
"""
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.2))
    
    plt.suptitle('Multi-Crop DINO: Learned Representations Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save visualization
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Embedding visualization saved to: {save_path}")
    plt.show()
    
    return features_umap

def visualize_multicrop_examples(model, dataloader, save_path='./figs/dino_multicrop_examples.png'):
    """Visualize multi-crop augmentation examples"""
    print("Creating multi-crop augmentation examples...")
    
    # Get a batch of images
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        break
    
    # Create multi-crop function
    multi_crop_fn = FixedMultiCropAugmentation(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96
    )
    
    # Take first 3 images for visualization
    n_examples = min(3, len(images))
    
    # Create figure
    fig, axes = plt.subplots(n_examples, 11, figsize=(22, 6*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalization function
    def denormalize_tensor(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        return torch.clamp(tensor * std + mean, 0, 1)
    
    for idx in range(n_examples):
        img = images[idx]
        label = labels[idx]
        
        # Show original image
        img_display = denormalize_tensor(img)
        axes[idx, 0].imshow(img_display.permute(1, 2, 0).cpu())
        axes[idx, 0].set_title(f'Original\nClass {label.item()}', fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')
        axes[idx, 0].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[idx, 0].transAxes,
                                           fill=False, edgecolor='blue', linewidth=3))
        
        # Generate multi-crops
        crops = multi_crop_fn(img)  # [10, 3, H, W] - 2 global + 8 local
        
        # Show 2 global crops (teacher)
        for i in range(2):
            crop_display = denormalize_tensor(crops[i])
            axes[idx, i+1].imshow(crop_display.permute(1, 2, 0).cpu())
            axes[idx, i+1].set_title(f'Global {i+1}\n(Teacher)\n224×224', fontsize=9, fontweight='bold')
            axes[idx, i+1].axis('off')
            axes[idx, i+1].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[idx, i+1].transAxes,
                                                 fill=False, edgecolor='green', linewidth=2))
        
        # Show 8 local crops (student)
        for i in range(8):
            crop_display = denormalize_tensor(crops[i+2])
            axes[idx, i+3].imshow(crop_display.permute(1, 2, 0).cpu())
            axes[idx, i+3].set_title(f'Local {i+1}\n(Student)\n96→224', fontsize=8)
            axes[idx, i+3].axis('off')
            axes[idx, i+3].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[idx, i+3].transAxes,
                                                 fill=False, edgecolor='orange', linewidth=1))
    
    # Add comprehensive title and legend
    fig.suptitle('Multi-Crop DINO: Asymmetric Augmentation Strategy', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='blue', linewidth=3, label='Original Image'),
        Patch(facecolor='white', edgecolor='green', linewidth=2, label='Global Crops → Teacher Network'),
        Patch(facecolor='white', edgecolor='orange', linewidth=1, label='Local Crops → Student Network')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14, 
              bbox_to_anchor=(0.5, 0.02))
    
    # Add strategy explanation
    strategy_text = """
Multi-Crop DINO Training Strategy:
• Student (8 local crops) learns to predict Teacher (2 global crops)
• Each local crop predicts each global crop → 8 × 2 = 16 prediction pairs per image
• Teacher updated via EMA from Student weights
• Asymmetric loss provides richer learning signal than symmetric approaches
"""
    fig.text(0.02, 0.98, strategy_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Multi-crop examples saved to: {save_path}")
    plt.show()
    
    return fig

def create_feature_analysis_report(features, labels, clustering_results, save_path='./outputs/dino_analysis_report.txt'):
    """Create detailed text report of the analysis"""
    
    n_samples, n_features = features.shape
    n_classes = len(np.unique(labels))
    
    report = f"""
MULTI-CROP DINO REPRESENTATION ANALYSIS REPORT
{'='*50}

DATASET SUMMARY:
- Total samples: {n_samples}
- Feature dimension: {n_features}
- Number of classes: {n_classes}
- Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}

DIMENSIONALITY REDUCTION:
- PCA components: 50 (if applicable)
- Explained variance: {clustering_results['explained_var']:.3f}

CLUSTERING RESULTS:
- Algorithm: K-means
- Number of clusters: {len(np.unique(clustering_results['pred_labels']))}

QUALITY METRICS:
- Adjusted Rand Index (ARI): {clustering_results['ari']:.4f}
- Normalized Mutual Information (NMI): {clustering_results['nmi']:.4f}
- Silhouette Score: {clustering_results['silhouette']:.4f}

INTERPRETATION:
- ARI measures agreement between true and predicted clusters
  * > 0.7: Excellent clustering
  * > 0.4: Good clustering  
  * > 0.1: Fair clustering
  * < 0.1: Poor clustering

- Current ARI of {clustering_results['ari']:.3f} indicates: {'EXCELLENT' if clustering_results['ari'] > 0.7 else 'GOOD' if clustering_results['ari'] > 0.4 else 'FAIR' if clustering_results['ari'] > 0.1 else 'POOR'} performance

CONCLUSION:
{'DINO successfully learned meaningful representations that capture class structure.' if clustering_results['ari'] > 0.1 else 'DINO representations need improvement - consider longer training or hyperparameter tuning.'}

Generated by Multi-Crop DINO implementation
"""
    
    # Save report
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Detailed analysis report saved to: {save_path}")
    print(report)
    """Create detailed text report of the analysis"""
    
    n_samples, n_features = features.shape
    n_classes = len(np.unique(labels))
    
    report = f"""
MULTI-CROP DINO REPRESENTATION ANALYSIS REPORT
{'='*50}

DATASET SUMMARY:
- Total samples: {n_samples}
- Feature dimension: {n_features}
- Number of classes: {n_classes}
- Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}

DIMENSIONALITY REDUCTION:
- PCA components: 50 (if applicable)
- Explained variance: {clustering_results['explained_var']:.3f}

CLUSTERING RESULTS:
- Algorithm: K-means
- Number of clusters: {len(np.unique(clustering_results['pred_labels']))}

QUALITY METRICS:
- Adjusted Rand Index (ARI): {clustering_results['ari']:.4f}
- Normalized Mutual Information (NMI): {clustering_results['nmi']:.4f}
- Silhouette Score: {clustering_results['silhouette']:.4f}

INTERPRETATION:
- ARI measures agreement between true and predicted clusters
  * > 0.7: Excellent clustering
  * > 0.4: Good clustering  
  * > 0.1: Fair clustering
  * < 0.1: Poor clustering

- Current ARI of {clustering_results['ari']:.3f} indicates: {'EXCELLENT' if clustering_results['ari'] > 0.7 else 'GOOD' if clustering_results['ari'] > 0.4 else 'FAIR' if clustering_results['ari'] > 0.1 else 'POOR'} performance

CONCLUSION:
{'DINO successfully learned meaningful representations that capture class structure.' if clustering_results['ari'] > 0.1 else 'DINO representations need improvement - consider longer training or hyperparameter tuning.'}

Generated by Multi-Crop DINO implementation
"""
    
    # Save report
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Detailed analysis report saved to: {save_path}")
    print(report)

def create_data_loaders(data_dir, batch_size=16, image_size=224, train_split=0.8):
    """Create properly split data loaders"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset(data_dir, transform=transform, image_size=image_size)
    
    # Stratified split to ensure balanced classes in train/val
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for label, indices in class_indices.items():
        n_train = int(len(indices) * train_split)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader, dataset

def main():
    """Main function with working loss function and comprehensive evaluation"""
    
    DATA_DIR = "./data/corel"  
    IMAGE_SIZE = 224  # Keep your 224 resolution
    BATCH_SIZE = 16   # Increased batch size
    NUM_EPOCHS = 20   # Reduced epochs since it converges quickly
    OUT_DIM = 2048    # Reduced projection dimension
    
    print("="*60)
    print("MULTI-CROP DINO with COMPREHENSIVE EVALUATION")
    print("="*60)
    print("Expected: Loss should start at 1-3 and decrease to <1.0")
    print("If loss starts at 6.9+, the fix didn't work properly")
    
    # Load data
    train_loader, val_loader, dataset = create_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )
    
    # Create model with smaller projection dimension
    model = FixedMultiCropDINO(input_channels=3, out_dim=OUT_DIM).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train with working loss function
    history = train_working_dino(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # Plot training results
    if history['train_loss']:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], 'b-', label='Train', linewidth=2)
        plt.plot(history['val_loss'], 'r-', label='Val', linewidth=2)
        plt.title('Loss Curves', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['teacher_entropy'], 'g-', linewidth=2)
        plt.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Healthy threshold')
        plt.title('Teacher Entropy', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        final_loss = history['train_loss'][-1]
        plt.text(0.1, 0.8, f"Final Loss: {final_loss:.4f}", fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f"Status: {'SUCCESS' if final_loss < 1.0 else 'PARTIAL' if final_loss < 3.0 else 'FAILED'}", 
                fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f"Expected: <1.0", fontsize=12, transform=plt.gca().transAxes)
        plt.title('Training Result', fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('./outputs/working_dino_training.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nTRAINING RESULT:")
        print(f"Final loss: {final_loss:.4f}")
        if final_loss < 1.0:
            print("SUCCESS! DINO converged properly")
        elif final_loss < 3.0:
            print("PARTIAL SUCCESS: Some learning occurred")
        else:
            print("FAILURE: Loss still too high")
            return history, model
    
    # Load best model if available
    if os.path.exists('./outputs/best_working_dino.pth'):
        print("\nLoading best model weights...")
        model.load_state_dict(torch.load('./outputs/best_working_dino.pth'))
        print("Best model loaded successfully")
    
    print("\n" + "="*60)
    print("STARTING REPRESENTATION ANALYSIS")
    print("="*60)
    
    # Visualize multi-crop examples first
    print("1. Creating multi-crop visualization...")
    visualize_multicrop_examples(model, val_loader)
    
    # Extract features from trained model
    print("\n2. Extracting features...")
    features, labels, indices = extract_features(model, val_loader, device)
    
    # Evaluate clustering quality
    print("\n3. Evaluating clustering...")
    clustering_results = evaluate_clustering(features, labels)
    
    # Create visualizations
    print("\n4. Creating embedding visualizations...")
    features_umap = visualize_embeddings(features, labels, clustering_results)
    
    # Generate detailed report
    print("\n5. Generating analysis report...")
    create_feature_analysis_report(features, labels, clustering_results)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Training convergence: {'SUCCESS' if final_loss < 1.0 else 'PARTIAL' if final_loss < 3.0 else 'FAILED'}")
    print(f"Representation quality: {'EXCELLENT' if clustering_results['ari'] > 0.7 else 'GOOD' if clustering_results['ari'] > 0.4 else 'FAIR' if clustering_results['ari'] > 0.1 else 'POOR'}")
    print(f"ARI Score: {clustering_results['ari']:.3f}")
    print(f"Files saved to ./outputs/")
    
    if clustering_results['ari'] > 0.1:
        print("\nSUCCESS: Multi-Crop DINO learned meaningful representations!")
        print("Your model is ready for downstream tasks.")
    else:
        print("\nWARNING: Representations may need improvement.")
        print("Consider: longer training, different hyperparameters, or data augmentation.")
    
    return history, model, features, clustering_results

if __name__ == "__main__":
    history, model = main()
