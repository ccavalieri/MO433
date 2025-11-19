#!/usr/bin/env python
# coding: utf-8

# BYOL (Bootstrap Your Own Latent) - Implementation
# 
# BYOL learns visual representations without negative samples:
# - Uses separate online and target networks (both have encoder + projector)
# - Online network has additional predictor head
# - Target network updated via EMA from online network (entire network)
# - Uses MSE loss between online predictions and target projections
# - Avoids representation collapse via momentum updates and predictor asymmetry
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

# Helper functions to create network components
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

def create_projector(feature_dim, projection_dim=64):
    """Create projector architecture"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(True),
        nn.Linear(256, projection_dim)
    )

def create_predictor(projection_dim=64):
    """Create predictor architecture (only for online network)"""
    return nn.Sequential(
        nn.Linear(projection_dim, projection_dim//2),
        nn.BatchNorm1d(projection_dim//2),
        nn.ReLU(True),
        nn.Linear(projection_dim//2, projection_dim)
    )

class BYOLModel(nn.Module):
    def __init__(self, input_channels=1, projection_dim=64):
        super().__init__()
        
        # Calculate feature dimension after encoder: 64 * 2 * 2 = 256
        self.feature_dim = 64 * 2 * 2
        
        # ONLINE NETWORK (trainable)
        self.online_encoder = create_encoder(input_channels)
        self.online_projector = create_projector(self.feature_dim, projection_dim)
        self.predictor = create_predictor(projection_dim)
        
        # TARGET NETWORK (updated via EMA, no gradients)
        self.target_encoder = create_encoder(input_channels)
        self.target_projector = create_projector(self.feature_dim, projection_dim)
        
        # Initialize target network with online weights
        self._initialize_target_network()
        
        # Disable gradients for entire target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        print("Corrected BYOL Model initialized:")
        print(f"  - Online: Encoder + Projector + Predictor (trainable)")
        print(f"  - Target: Encoder + Projector (EMA updated)")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Projection dimension: {projection_dim}")
    
    def _initialize_target_network(self):
        """Initialize target network with online network weights"""
        # Copy encoder weights
        for online_param, target_param in zip(self.online_encoder.parameters(), 
                                            self.target_encoder.parameters()):
            target_param.data.copy_(online_param.data)
        
        # Copy projector weights
        for online_param, target_param in zip(self.online_projector.parameters(), 
                                            self.target_projector.parameters()):
            target_param.data.copy_(online_param.data)
    
    def update_target_network(self, momentum=0.996):
        """Update entire target network with exponential moving average"""
        # Update target encoder
        for online_param, target_param in zip(self.online_encoder.parameters(), 
                                            self.target_encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * online_param.data
        
        # Update target projector
        for online_param, target_param in zip(self.online_projector.parameters(), 
                                            self.target_projector.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * online_param.data
    
    def forward(self, x1, x2):
        """Forward pass through both online and target networks"""
        
        # ONLINE NETWORK (gradients enabled)
        online_features1 = self.online_encoder(x1)
        online_features2 = self.online_encoder(x2)
        
        online_proj1 = self.online_projector(online_features1)
        online_proj2 = self.online_projector(online_features2)
        
        online_pred1 = self.predictor(online_proj1)
        online_pred2 = self.predictor(online_proj2)
        
        # TARGET NETWORK (no gradients)
        with torch.no_grad():
            target_features1 = self.target_encoder(x1)
            target_features2 = self.target_encoder(x2)
            
            target_proj1 = self.target_projector(target_features1)
            target_proj2 = self.target_projector(target_features2)
        
        return online_pred1, online_pred2, target_proj1, target_proj2, online_features1
    
    def encode_only(self, x):
        """Extract online encoder features only (for representation learning evaluation)"""
        features = self.online_encoder(x)
        return features

# Create model with corrected BYOL architecture
model = BYOLModel(input_channels=1, projection_dim=64).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
target_params = sum(p.numel() for p in model.target_encoder.parameters()) + sum(p.numel() for p in model.target_projector.parameters())

print(f"\nModel Parameters:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters (online): {trainable_params:,}")
print(f"  Target parameters (EMA): {target_params:,}")

# BYOL Data Augmentation
def get_byol_augmentation():
    """Augmentation pipeline for BYOL"""
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

def create_byol_pairs(batch, augment_fn):
    """Create augmented pairs for BYOL training"""
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

# BYOL Loss Function
def byol_loss(online_pred1, online_pred2, target_proj1, target_proj2):
    """
    BYOL loss function - MSE between online predictions and target projections
    
    Args:
        online_pred1: Online network predictions for first augmented batch
        online_pred2: Online network predictions for second augmented batch  
        target_proj1: Target network projections for first augmented batch
        target_proj2: Target network projections for second augmented batch
    """
    # Normalize predictions and projections
    online_pred1 = F.normalize(online_pred1, dim=1)
    online_pred2 = F.normalize(online_pred2, dim=1) 
    target_proj1 = F.normalize(target_proj1, dim=1)
    target_proj2 = F.normalize(target_proj2, dim=1)
    
    # BYOL loss: predict target2 from online1, and target1 from online2
    loss1 = 2 - 2 * (online_pred1 * target_proj2).sum(dim=1).mean()
    loss2 = 2 - 2 * (online_pred2 * target_proj1).sum(dim=1).mean()
    
    # Total loss is the mean of both losses
    total_loss = (loss1 + loss2) / 2
    
    return total_loss

# Training Functions
def train_batch_byol(data, model, optimizer, augment_fn, momentum=0.996):
    model.train()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_byol_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass through both online and target networks
    online_pred1, online_pred2, target_proj1, target_proj2, _ = model(aug1, aug2)
    
    # Compute BYOL loss
    loss = byol_loss(online_pred1, online_pred2, target_proj1, target_proj2)
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Update entire target network with momentum
    model.update_target_network(momentum)
    
    return loss

@torch.no_grad()
def validate_batch_byol(data, model, augment_fn):
    model.eval()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_byol_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    # Forward pass
    online_pred1, online_pred2, target_proj1, target_proj2, _ = model(aug1, aug2)
    
    # Compute loss
    loss = byol_loss(online_pred1, online_pred2, target_proj1, target_proj2)
    
    return loss

# Model Setup and Training
model = BYOLModel(input_channels=1, projection_dim=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Get augmentation function
augment_fn = get_byol_augmentation()

num_epochs = 12
log = Report(num_epochs)

print(f"\nStarting Corrected BYOL training on {device}...")
print(f"Training samples: {len(trn_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print("="*60)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Momentum schedule (starts at 0.99, increases to 0.999)
    momentum = 0.99 + (0.999 - 0.99) * (epoch / num_epochs)
    
    # Training
    epoch_losses = []
    N = len(trn_dl)
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch_byol(data, model, optimizer, augment_fn, momentum)
        epoch_losses.append(loss.item())
        log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end='\r')
    
    avg_train_loss = np.mean(epoch_losses)
    print(f"  Average training loss: {avg_train_loss:.4f}")
    
    # Validation
    val_losses = []
    N = len(val_dl)
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch_byol(data, model, augment_fn)
        val_losses.append(loss.item())
        log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end='\r')
    
    avg_val_loss = np.mean(val_losses)
    print(f"  Average validation loss: {avg_val_loss:.4f}")
    print(f"  Target momentum: {momentum:.4f}")
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Current learning rate: {current_lr:.2e}")
    
    # Early convergence check for BYOL
    if avg_train_loss < 0.5 and epoch >= 2:
        print(f"  ✓ Good BYOL convergence achieved at epoch {epoch+1}")
    
    log.report_avgs(epoch+1)

print("\n" + "="*60)
print("CORRECTED BYOL TRAINING COMPLETED!")
print("="*60)
print("Architecture Summary:")
print("✓ Online Network: Encoder + Projector + Predictor (trainable)")
print("✓ Target Network: Encoder + Projector (EMA updated)")
print("✓ Target updated via momentum from online (entire network)")
print("✓ No shared weights between online and target")
print("✓ Proper asymmetric predictor architecture")
print("="*60)

# Plot Training Progress
log.plot_epochs(log=True)

# Sample Augmentations Visualization
print("\nVisualizing BYOL augmentations...")
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

plt.suptitle('BYOL Data Augmentations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./figs/byol_augmentations.png', dpi=300, bbox_inches='tight')
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
            
            # Latent features from online encoder (not projection head)
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

def plot_umap_comparison(embeddings_dict, labels, latent_dim, save_path='./figs/byol_umap_comparison.png'):
    """Plot and save comparison of three UMAP projections"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['Original Data (784D)', f'BYOL Encoder Features ({latent_dim}D)', 'PCA Reduced (50D)']
    
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
    fig.suptitle('UMAP Projections Comparison: BYOL vs PCA', 
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
print("UMAP PROJECTIONS COMPARISON: BYOL ANALYSIS")
print("="*60)

# Extract all data types
raw_data, latent_features, labels = extract_data_for_comparison(model, val_dl, max_samples=2000)

# Determine latent dimension
latent_dim = latent_features.shape[1]
print(f"BYOL online encoder latent dimension: {latent_dim}")
print(f"Using PCA with fixed 50 dimensions for fair comparison")

# Apply PCA to raw data (fixed at 50 dimensions)
pca_features = apply_pca_reduction(raw_data, n_components=50)

# Apply UMAP to all three representations
print("\n" + "="*40)
print("APPLYING UMAP PROJECTIONS")
print("="*40)

embeddings = {}
embeddings['Original Data (784D)'] = apply_umap_projection(raw_data, 'Original Data')
embeddings[f'BYOL Encoder Features ({latent_dim}D)'] = apply_umap_projection(latent_features, 'BYOL Encoder Features')
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
print(f"• BYOL Encoder Features ({latent_dim}D): Self-supervised features from ONLINE encoder")
print("• PCA Reduced (50D): Linear dimensionality reduction preserving variance")
print("\nHigher metrics indicate better separation of digit classes.")
print("BYOL learns representations via separate online-target networks,")
print("with target updated via EMA from online (entire network, not just projector).")

print(f"\nVisualization saved as: ./figs/byol_umap_comparison.png")

# t-SNE Comparison with BYOL Features
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
plt.title('t-SNE Projection of BYOL Encoder Features', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('./figs/byol_tsne.png', dpi=300, bbox_inches='tight')
plt.show()

print("t-SNE visualization saved as: ./figs/byol_tsne.png")

print("\n" + "="*50)
print("BYOL TRAINING COMPLETE!")
print("="*50)
print("BYOL has learned representations through proper self-supervised learning:")
print("• Separate online and target networks (encoder + projector each)")
print("• Target updated via EMA from online (entire network)")
print("• Only online network has predictor (asymmetric architecture)")
print("• No negative samples needed - avoids collapse via momentum + predictor asymmetry")
print("• Online encoder learns meaningful representations for downstream tasks")
print("• More stable than contrastive methods - excellent for representation learning")
print("="*50)
