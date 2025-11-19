#!/usr/bin/env python
# coding: utf-8

# SimCLR (Simple Contrastive Learning of Visual Representations) Implementation
# 
# SimCLR learns visual representations through contrastive learning:
# - Creates augmented pairs from the same image (positive pairs)
# - Uses InfoNCE loss to make positive pairs similar and negative pairs dissimilar
# - No need for image reconstruction - focuses purely on representation learning
#

# In[1]:


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


# In[2]: SimCLR Model Architecture


class SimCLREncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder (same as before but no decoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        
        # Calculate the size after convolutions
        # This will give us the flattened size: 64 * 2 * 2 = 256
        self.feature_dim = 64 * 2 * 2
        
        # Projection head for SimCLR (maps to lower dimensional space for contrastive learning)
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        # Get features from encoder
        features = self.encoder(x)
        # Apply projection head
        projections = self.projection_head(features)
        return features, projections
    
    def encode_only(self, x):
        """Extract encoder features only (for representation learning evaluation)"""
        features = self.encoder(x)
        return features

model = SimCLREncoder(latent_dim=128).to(device)
from torchsummary import summary
summary(model, torch.zeros(2,1,28,28));


# In[3]: SimCLR Data Augmentation


def get_simclr_augmentation():
    """Augmentation pipeline for SimCLR"""
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

def create_simclr_pairs(batch, augment_fn):
    """Create augmented pairs for SimCLR training"""
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


# In[4]: SimCLR Loss Function (InfoNCE/NT-Xent)


def simclr_loss(projections_1, projections_2, temperature=0.5):
    """
    SimCLR loss function (InfoNCE/NT-Xent)
    
    Args:
        projections_1: Projections from first augmented batch [batch_size, projection_dim]
        projections_2: Projections from second augmented batch [batch_size, projection_dim]
        temperature: Temperature parameter for softmax
    """
    batch_size = projections_1.shape[0]
    
    # Normalize projections
    projections_1 = F.normalize(projections_1, dim=1)
    projections_2 = F.normalize(projections_2, dim=1)
    
    # Concatenate projections: [2*batch_size, projection_dim]
    projections = torch.cat([projections_1, projections_2], dim=0)
    
    # Compute similarity matrix: [2*batch_size, 2*batch_size]
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    
    # Create labels for positive pairs
    # For SimCLR: (i, i+batch_size) and (i+batch_size, i) are positive pairs
    labels = torch.cat([torch.arange(batch_size, 2*batch_size), 
                       torch.arange(0, batch_size)]).to(device)
    
    # Mask to remove self-similarity (diagonal)
    mask = torch.eye(2*batch_size).bool().to(device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


# In[5]: Training Functions


def train_batch_simclr(data, model, optimizer, augment_fn, temperature=0.5):
    model.train()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_simclr_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass through both augmented versions
    _, proj1 = model(aug1)
    _, proj2 = model(aug2)
    
    # Compute SimCLR loss
    loss = simclr_loss(proj1, proj2, temperature)
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss

@torch.no_grad()
def validate_batch_simclr(data, model, augment_fn, temperature=0.5):
    model.eval()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_simclr_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    # Forward pass
    _, proj1 = model(aug1)
    _, proj2 = model(aug2)
    
    # Compute loss
    loss = simclr_loss(proj1, proj2, temperature)
    
    return loss


# In[6]: Model Setup and Training


model = SimCLREncoder(latent_dim=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)  # Increased LR
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Simpler scheduler

# Get augmentation function
augment_fn = get_simclr_augmentation()

num_epochs = 7  # Reduced epochs for MNIST
log = Report(num_epochs)

print(f"Starting SimCLR training on {device}...")
print(f"Training samples: {len(trn_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Training
    epoch_losses = []
    N = len(trn_dl)
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch_simclr(data, model, optimizer, augment_fn)
        epoch_losses.append(loss.item())
        log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end='\r')
    
    avg_train_loss = np.mean(epoch_losses)
    print(f"  Average training loss: {avg_train_loss:.4f}")
    
    # Validation
    val_losses = []
    N = len(val_dl)
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch_simclr(data, model, augment_fn)
        val_losses.append(loss.item())
        log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end='\r')
    
    avg_val_loss = np.mean(val_losses)
    print(f"  Average validation loss: {avg_val_loss:.4f}")
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Current learning rate: {current_lr:.2e}")
    
    # Early convergence check
    if avg_train_loss < 1.0 and epoch >= 2:
        print(f"  ✓ Good convergence achieved at epoch {epoch+1}")
    
    log.report_avgs(epoch+1)

print("SimCLR Training completed!")


# In[7]: Plot Training Progress


log.plot_epochs(log=True)


# In[8]: Sample Augmentations Visualization


print("\nVisualizing SimCLR augmentations...")
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

plt.suptitle('SimCLR Data Augmentations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./figs/simclr_augmentations.png', dpi=300, bbox_inches='tight')
plt.show()


# In[9]: UMAP Projection Comparison Functions


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

def plot_umap_comparison(embeddings_dict, labels, latent_dim, save_path='./figs/simclr_umap_comparison.png'):
    """Plot and save comparison of three UMAP projections"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['Original Data (784D)', f'SimCLR Encoder Features ({latent_dim}D)', 'PCA Reduced (50D)']
    
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
    fig.suptitle('UMAP Projections Comparison: SimCLR vs PCA', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"UMAP comparison plot saved as: {save_path}")
    
    # Show the plot
    plt.show()
    
    return fig


# In[10]: Main UMAP Analysis


print("\n" + "="*60)
print("UMAP PROJECTIONS COMPARISON: SIMCLR ANALYSIS")
print("="*60)

# Extract all data types
raw_data, latent_features, labels = extract_data_for_comparison(model, val_dl, max_samples=2000)

# Determine latent dimension
latent_dim = latent_features.shape[1]
print(f"SimCLR encoder latent dimension: {latent_dim}")
print(f"Using PCA with fixed 50 dimensions for fair comparison")

# Apply PCA to raw data (fixed at 50 dimensions)
pca_features = apply_pca_reduction(raw_data, n_components=50)

# Apply UMAP to all three representations
print("\n" + "="*40)
print("APPLYING UMAP PROJECTIONS")
print("="*40)

embeddings = {}
embeddings['Original Data (784D)'] = apply_umap_projection(raw_data, 'Original Data')
embeddings[f'SimCLR Encoder Features ({latent_dim}D)'] = apply_umap_projection(latent_features, 'SimCLR Encoder Features')
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
print(f"• SimCLR Encoder Features ({latent_dim}D): Contrastive learned features optimized for representation")
print("• PCA Reduced (50D): Linear dimensionality reduction preserving variance")
print("\nHigher metrics indicate better separation of digit classes.")
print("SimCLR should learn discriminative features through contrastive learning,")
print("potentially outperforming both raw pixels and PCA for clustering tasks.")

print(f"\nVisualization saved as: ./figs/simclr_umap_comparison.png")


# In[11]: t-SNE Comparison with SimCLR Features


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
plt.title('t-SNE Projection of SimCLR Encoder Features', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('./figs/simclr_tsne.png', dpi=300, bbox_inches='tight')
plt.show()

print("t-SNE visualization saved as: ./figs/simclr_tsne.png")

print("\n" + "="*50)
print("SIMCLR TRAINING COMPLETE!")
print("="*50)
print("SimCLR has learned representations through contrastive learning:")
print("• No reconstruction needed - focuses purely on learning good features")
print("• Uses data augmentation and InfoNCE loss")
print("• Should show better clustering than traditional autoencoders")
print("• Encoder features are optimized for downstream tasks")
print("="*50)


# In[ ]:
