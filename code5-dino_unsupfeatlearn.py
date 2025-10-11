#!/usr/bin/env python
# coding: utf-8

# DINO (Self-Distillation with No Labels) - Standard Implementation
# 
# DINO learns visual representations through self-distillation:
# - Uses separate teacher-student networks (encoder + projection head each)
# - Student learns to predict teacher's outputs for different augmented views
# - Teacher updated via EMA from student (entire network)
# - Uses cross-entropy loss with temperature scaling and centering
# - Prevents collapse via centering mechanism instead of negative samples
#

import torch
from torch import nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch_snippets import *
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_snippets.torch_loader import Report
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Base transform for data loading (no augmentation here)
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

trn_ds = MNIST('./data/', transform=base_transform, train=True, download=True)
val_ds = MNIST('./data/', transform=base_transform, train=False, download=True)

batch_size = 128
trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


# Helper function to create encoder
def create_encoder(input_channels=1):
    """Create encoder architecture"""
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, 3, stride=3, padding=1), 
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), 
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.MaxPool2d(2, stride=1)
    )

def create_projection_head(feature_dim, projection_dim=256):
    """Create projection head architecture"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_dim, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Linear(512, projection_dim, bias=False)
    )


class DINOModel(nn.Module):
    def __init__(self, input_channels=1, projection_dim=256):
        super().__init__()
        
        # Calculate feature dimension after encoder
        # For MNIST: 64 * 2 * 2 = 256
        self.feature_dim = 64 * 2 * 2
        
        # STUDENT NETWORK (trainable)
        self.student_encoder = create_encoder(input_channels)
        self.student_head = create_projection_head(self.feature_dim, projection_dim)
        
        # TEACHER NETWORK (updated via EMA, no gradients)
        self.teacher_encoder = create_encoder(input_channels)
        self.teacher_head = create_projection_head(self.feature_dim, projection_dim)
        
        # Initialize teacher network with student weights
        self._initialize_teacher_network()
        
        # Disable gradients for entire teacher network
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        
        # DINO centering mechanism
        self.center = torch.zeros(1, projection_dim)
        
        print("✓ Standard DINO Model initialized:")
        print(f"  - Student: Encoder + Projection Head (trainable)")
        print(f"  - Teacher: Encoder + Projection Head (EMA updated)")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Projection dimension: {projection_dim}")
    
    def _initialize_teacher_network(self):
        """Initialize teacher network with student network weights"""
        # Copy encoder weights
        for student_param, teacher_param in zip(self.student_encoder.parameters(), 
                                              self.teacher_encoder.parameters()):
            teacher_param.data.copy_(student_param.data)
        
        # Copy projection head weights
        for student_param, teacher_param in zip(self.student_head.parameters(), 
                                              self.teacher_head.parameters()):
            teacher_param.data.copy_(student_param.data)
    
    def update_teacher_network(self, momentum=0.996):
        """Update entire teacher network with exponential moving average"""
        # Update teacher encoder
        for student_param, teacher_param in zip(self.student_encoder.parameters(), 
                                              self.teacher_encoder.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
        
        # Update teacher projection head
        for student_param, teacher_param in zip(self.student_head.parameters(), 
                                              self.teacher_head.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
    
    def update_center(self, teacher_output):
        """Update center used for teacher output centering"""
        self.center = self.center.to(teacher_output.device)
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) / len(teacher_output)
        # Update center with momentum
        self.center = self.center * 0.9 + batch_center * 0.1
    
    def forward(self, x1, x2):
        """Forward pass through both student and teacher networks"""
        
        # STUDENT NETWORK (gradients enabled)
        student_features1 = self.student_encoder(x1)
        student_features2 = self.student_encoder(x2)
        student_out1 = self.student_head(student_features1)
        student_out2 = self.student_head(student_features2)
        
        # TEACHER NETWORK (no gradients)
        with torch.no_grad():
            teacher_features1 = self.teacher_encoder(x1)
            teacher_features2 = self.teacher_encoder(x2)
            teacher_out1 = self.teacher_head(teacher_features1)
            teacher_out2 = self.teacher_head(teacher_features2)
        
        return student_out1, student_out2, teacher_out1, teacher_out2, student_features1
    
    def encode_only(self, x):
        """Extract student encoder features only (for representation learning evaluation)"""
        features = self.student_encoder(x)
        return features

# Create model with standard DINO architecture
model = DINOModel(input_channels=1, projection_dim=256).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
teacher_params = sum(p.numel() for p in model.teacher_encoder.parameters()) + sum(p.numel() for p in model.teacher_head.parameters())

print(f"\nModel Parameters:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters (student): {trainable_params:,}")
print(f"  Teacher parameters (EMA): {teacher_params:,}")


# DINO Data Augmentation
def get_dino_augmentation():
    """Augmentation pipeline for DINO"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Add some noise
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize([0.5], [0.5])
    ])

def create_dino_pairs(batch, augment_fn):
    """Create augmented pairs for DINO training"""
    batch_size = batch.shape[0]
    
    # Create two augmented versions of each image
    augmented_1 = []
    augmented_2 = []
    
    for img in batch:
        # Convert to proper format for augmentation
        img_denorm = (img * 0.5 + 0.5)  # Denormalize from [-1,1] to [0,1]
        
        # Create two different augmented versions
        aug1 = augment_fn(img_denorm[0])  # Remove channel dimension for PIL
        aug2 = augment_fn(img_denorm[0])
        
        augmented_1.append(aug1)
        augmented_2.append(aug2)
    
    augmented_1 = torch.stack(augmented_1)
    augmented_2 = torch.stack(augmented_2)
    
    return augmented_1, augmented_2


# DINO Loss Function
def dino_loss(student_out1, student_out2, teacher_out1, teacher_out2, center, 
              student_temp=0.1, teacher_temp=0.04):
    """
    DINO loss function - cross-entropy between student and teacher outputs
    
    Args:
        student_out1: Student network output for first augmented batch
        student_out2: Student network output for second augmented batch  
        teacher_out1: Teacher network output for first augmented batch
        teacher_out2: Teacher network output for second augmented batch
        center: Center for teacher output centering
        student_temp: Temperature for student softmax
        teacher_temp: Temperature for teacher softmax
    """
    # Student outputs (apply temperature)
    student_out1 = student_out1 / student_temp
    student_out2 = student_out2 / student_temp
    
    # Teacher outputs (apply centering and temperature)
    teacher_out1_centered = (teacher_out1 - center) / teacher_temp
    teacher_out2_centered = (teacher_out2 - center) / teacher_temp
    
    # Teacher targets (softmax, no gradients)
    teacher_target1 = F.softmax(teacher_out1_centered, dim=-1).detach()
    teacher_target2 = F.softmax(teacher_out2_centered, dim=-1).detach()
    
    # DINO loss: student predicts teacher targets for different views
    loss1 = -torch.sum(teacher_target2 * F.log_softmax(student_out1, dim=-1), dim=-1).mean()
    loss2 = -torch.sum(teacher_target1 * F.log_softmax(student_out2, dim=-1), dim=-1).mean()
    
    # Total loss is the mean of both cross-entropy losses
    total_loss = (loss1 + loss2) / 2
    
    return total_loss


# Training Functions
def train_batch_dino(data, model, optimizer, augment_fn, momentum=0.996):
    model.train()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_dino_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass through both student and teacher networks
    student_out1, student_out2, teacher_out1, teacher_out2, _ = model(aug1, aug2)
    
    # Update center with teacher outputs
    teacher_output = torch.cat([teacher_out1, teacher_out2])
    model.update_center(teacher_output)
    
    # Compute DINO loss
    loss = dino_loss(student_out1, student_out2, teacher_out1, teacher_out2, model.center)
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
    
    optimizer.step()
    
    # Update entire teacher network with momentum
    model.update_teacher_network(momentum)
    
    return loss

@torch.no_grad()
def validate_batch_dino(data, model, augment_fn):
    model.eval()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_dino_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    # Forward pass
    student_out1, student_out2, teacher_out1, teacher_out2, _ = model(aug1, aug2)
    
    # Compute loss (don't update center during validation)
    loss = dino_loss(student_out1, student_out2, teacher_out1, teacher_out2, model.center)
    
    return loss


# Model Setup and Training
model = DINOModel(input_channels=1, projection_dim=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.04)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# Get augmentation function
augment_fn = get_dino_augmentation()

num_epochs = 8
log = Report(num_epochs)

print(f"\nStarting Standard DINO training on {device}...")
print(f"Training samples: {len(trn_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print("="*60)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Momentum schedule (starts at 0.996, increases to 0.999)
    momentum = 0.996 + (0.999 - 0.996) * (epoch / num_epochs)
    
    # Training
    epoch_losses = []
    N = len(trn_dl)
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch_dino(data, model, optimizer, augment_fn, momentum)
        epoch_losses.append(loss.item())
        log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end='\r')
    
    avg_train_loss = np.mean(epoch_losses)
    print(f"  Average training loss: {avg_train_loss:.4f}")
    
    # Validation
    val_losses = []
    N = len(val_dl)
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch_dino(data, model, augment_fn)
        val_losses.append(loss.item())
        log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end='\r')
    
    avg_val_loss = np.mean(val_losses)
    print(f"  Average validation loss: {avg_val_loss:.4f}")
    print(f"  Teacher momentum: {momentum:.4f}")
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Current learning rate: {current_lr:.2e}")
    
    # Early convergence check for DINO
    if avg_train_loss < 2.0 and epoch >= 2:
        print(f"  ✓ Good DINO convergence achieved at epoch {epoch+1}")
    
    log.report_avgs(epoch+1)

print("\n" + "="*60)
print("STANDARD DINO TRAINING COMPLETED!")
print("="*60)
print("Architecture Summary:")
print("✓ Student Network: Separate encoder + projection head (trainable)")
print("✓ Teacher Network: Separate encoder + projection head (EMA updated)")
print("✓ Teacher updated via momentum from student (entire network)")
print("✓ No shared weights between teacher and student")
print("✓ Proper teacher-student knowledge distillation dynamics")
print("="*60)


# Plot Training Progress
log.plot_epochs(log=True)


# Sample Augmentations Visualization
print("\nVisualizing DINO augmentations...")
sample_data, _ = next(iter(val_dl))
sample_img = sample_data[0:1]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

# Original image
original = (sample_img[0, 0] * 0.5 + 0.5)  # Denormalize and remove channel dim
axes[0].imshow(original.numpy(), cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

# Show 9 augmented versions
for i in range(1, 10):
    aug_img = augment_fn((sample_img[0, 0] * 0.5 + 0.5))
    # Remove channel dimension and denormalize for display
    if len(aug_img.shape) == 3:  # If shape is (1, 28, 28)
        aug_display = (aug_img[0] * 0.5 + 0.5)  # Remove channel dim and denormalize
    else:  # If shape is already (28, 28)
        aug_display = (aug_img * 0.5 + 0.5)  # Just denormalize
    
    axes[i].imshow(aug_display.numpy(), cmap='gray')
    axes[i].set_title(f'Aug {i}')
    axes[i].axis('off')

plt.suptitle('DINO Data Augmentations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./figs/dino_augmentations.png', dpi=300, bbox_inches='tight')
plt.show()


# UMAP Projection Comparison Functions
def extract_data_for_comparison(model, dataloader, max_samples=2000):
    """Extract raw data, latent features, and labels"""
    model.eval()
    
    raw_data = []
    latent_features = []
    labels = []
    
    print("Extracting data for UMAP comparison...")
    
    with torch.no_grad():
        samples_collected = 0
        for data, label in dataloader:
            if samples_collected >= max_samples:
                break
            
            data = data.to(device)
            
            # Raw pixel data
            raw_pixels = data.view(data.size(0), -1).cpu().numpy()
            raw_data.append(raw_pixels)
            
            # Latent features from encoder (not projection head)
            latent = model.encode_only(data)
            latent_flat = latent.view(latent.size(0), -1).cpu().numpy()
            latent_features.append(latent_flat)
            
            # Labels
            labels.append(label.numpy())
            
            samples_collected += len(data)
    
    raw_data = np.vstack(raw_data)[:max_samples]
    latent_features = np.vstack(latent_features)[:max_samples]
    labels = np.hstack(labels)[:max_samples]
    
    print(f"Extracted {len(raw_data)} samples:")
    print(f"  Raw data shape: {raw_data.shape}")
    print(f"  Latent features shape: {latent_features.shape}")
    
    return raw_data, latent_features, labels

def apply_pca_reduction(data, n_components=50):
    """Apply PCA to reduce dimensionality"""
    print(f"Applying PCA reduction from {data.shape[1]} to {n_components} dimensions...")
    
    # Standardize first
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data_scaled)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance ratio: {explained_variance:.3f}")
    
    return data_pca

def apply_umap_projection(features, method_name, n_neighbors=15, min_dist=0.1):
    """Apply UMAP to project features to 2D"""
    print(f"Applying UMAP projection for {method_name}...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        random_state=42,
        verbose=False
    )
    
    embedding = reducer.fit_transform(features_scaled)
    
    print(f"UMAP projection completed for {method_name}. Embedding shape: {embedding.shape}")
    return embedding

def calculate_metrics(embedding, labels, method_name):
    """Calculate clustering quality metrics"""
    # Silhouette score
    silhouette = silhouette_score(embedding, labels)
    
    # K-means clustering and ARI
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embedding)
    ari = adjusted_rand_score(labels, predicted_labels)
    
    print(f"{method_name}:")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    
    return {'silhouette': silhouette, 'ari': ari}

def plot_umap_comparison(embeddings_dict, labels, latent_dim, save_path='./figs/dino_umap_comparison.png'):
    """Plot and save comparison of three UMAP projections"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['Original Data (784D)', f'DINO Encoder Features ({latent_dim}D)', 'PCA Reduced (50D)']
    
    for i, method in enumerate(methods):
        embedding = embeddings_dict[method]
        
        # Create scatter plot
        scatter = axes[i].scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=labels, 
            cmap='tab10', 
            alpha=0.7, 
            s=15
        )
        
        # Set title and labels
        axes[i].set_title(method, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('UMAP Component 1')
        axes[i].set_ylabel('UMAP Component 2')
        axes[i].grid(True, alpha=0.3)
        
        # Remove tick labels for cleaner look
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax, ticks=range(10))
    cbar.set_label('MNIST Digit Classes', fontsize=12)
    cbar.set_ticklabels([str(i) for i in range(10)])
    
    # Add main title
    fig.suptitle('UMAP Projections Comparison: DINO vs PCA', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"UMAP comparison plot saved as: {save_path}")
    
    # Show the plot
    plt.show()
    
    return fig


# Main UMAP Analysis
print("\n" + "="*60)
print("UMAP PROJECTIONS COMPARISON: STANDARD DINO ANALYSIS")
print("="*60)

# Extract all data types
raw_data, latent_features, labels = extract_data_for_comparison(model, val_dl, max_samples=2000)

# Determine latent dimension
latent_dim = latent_features.shape[1]
print(f"DINO encoder latent dimension: {latent_dim}")
print(f"Using PCA with fixed 50 dimensions for fair comparison")

# Apply PCA to raw data (fixed at 50 dimensions)
pca_features = apply_pca_reduction(raw_data, n_components=50)

# Apply UMAP to all three representations
print("\n" + "="*40)
print("APPLYING UMAP PROJECTIONS")
print("="*40)

embeddings = {}
embeddings['Original Data (784D)'] = apply_umap_projection(raw_data, 'Original Data')
embeddings[f'DINO Encoder Features ({latent_dim}D)'] = apply_umap_projection(latent_features, 'DINO Encoder Features')
embeddings['PCA Reduced (50D)'] = apply_umap_projection(pca_features, 'PCA Reduced')

# Create comparison plot
print("\n" + "="*40)
print("CREATING COMPARISON VISUALIZATION")
print("="*40)

fig = plot_umap_comparison(embeddings, labels, latent_dim)

# Calculate and compare metrics
print("\n" + "="*40)
print("CLUSTERING QUALITY METRICS COMPARISON")
print("="*40)

metrics = {}
for method, embedding in embeddings.items():
    metrics[method] = calculate_metrics(embedding, labels, method)

# Summary comparison
print("\n" + "="*40)
print("SUMMARY COMPARISON")
print("="*40)

print(f"{'Method':<40} {'Silhouette':<12} {'ARI':<8}")
print("-" * 60)
for method, metric in metrics.items():
    print(f"{method:<40} {metric['silhouette']:<12.3f} {metric['ari']:<8.3f}")

# Find best method for each metric
best_silhouette = max(metrics.keys(), key=lambda x: metrics[x]['silhouette'])
best_ari = max(metrics.keys(), key=lambda x: metrics[x]['ari'])

print(f"\nBest cluster separation (Silhouette): {best_silhouette}")
print(f"Best label agreement (ARI): {best_ari}")

print("\n" + "="*40)
print("ANALYSIS INSIGHTS")
print("="*40)
print("• Original Data (784D): Raw pixel intensities, high-dimensional")
print(f"• DINO Encoder Features ({latent_dim}D): Self-distilled features from STUDENT encoder")
print("• PCA Reduced (50D): Linear dimensionality reduction preserving variance")
print("\nHigher metrics indicate better separation of digit classes.")
print("Standard DINO learns representations via separate teacher-student networks,")
print("with teacher updated via EMA from student (entire network, not just heads).")

print(f"\nVisualization saved as: ./figs/dino_umap_comparison.png")


# t-SNE Comparison with DINO Features
print("\n" + "="*40)
print("ADDITIONAL t-SNE VISUALIZATION")
print("="*40)

latent_vectors = []
classes = []

for im, clss in val_dl:
    im = im.to(device)
    latent_vectors.append(model.encode_only(im).view(len(im), -1))
    classes.extend(clss)

latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy()

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
clustered = tsne.fit_transform(latent_vectors)

fig = plt.figure(figsize=(12, 10))
cmap = plt.get_cmap('Spectral', 10)
plt.scatter(*zip(*clustered), c=classes, cmap=cmap, alpha=0.7, s=20)
plt.colorbar(drawedges=True)
plt.title('t-SNE Projection of Standard DINO Encoder Features', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('./figs/dino_tsne.png', dpi=300, bbox_inches='tight')
plt.show()

print("t-SNE visualization saved as: ./figs/dino_tsne.png")

print("\n" + "="*50)
print("STANDARD DINO ANALYSIS COMPLETE!")
print("="*50)
print("Standard DINO has learned representations through proper self-distillation:")
print("• Separate teacher-student networks (encoder + projection head each)")
print("• Teacher updated via EMA from student (entire network)")
print("• Cross-entropy loss with temperature scaling and centering")
print("• No shared weights - proper knowledge distillation dynamics")
print("• Student encoder learns meaningful representations")
print("• More stable than contrastive methods - excellent for representation learning")
print("="*50)