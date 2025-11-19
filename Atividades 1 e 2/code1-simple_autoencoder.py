#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sizhky/Modern-Computer-Vision-with-PyTorch/blob/master/Chapter11/simple_auto_encoder_with_different_latent_size.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import torch
from torch import nn as nn
from torch_snippets import *
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch_snippets.torch_loader import Report
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[2]:


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.to(device))
])


# In[3]:


trn_ds = MNIST('./data/', transform=img_transform, train=True, download=True)
val_ds = MNIST('./data/', transform=img_transform, train=False, download=True)


# In[4]:


batch_size = 256
trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


# In[5]:


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latend_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True), 
            #nn.Linear(64, 12),  nn.ReLU(True), 
            nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(
            #nn.Linear(latent_dim, 12), nn.ReLU(True),
            nn.Linear(latent_dim, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True), 
            nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = x.view(len(x), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(len(decoded), 1, 28, 28)
        return decoded
    
    def encode(self, x):
        """Extract latent features only"""
        x = x.view(len(x), -1)
        return self.encoder(x)


# In[6]:

latent_dim=3
from torchsummary import summary
model = AutoEncoder(latent_dim).to(device)
summary(model, torch.zeros(2,1,28,28));


# In[7]:


def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss


# In[8]:


model = AutoEncoder(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 5
log = Report(num_epochs)

for epoch in range(num_epochs):
    N = len(trn_dl)
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch(data, model, criterion, optimizer)
        log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end='\r')

    N = len(val_dl)
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch(data, model, criterion)
        log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end='\r')
    log.report_avgs(epoch+1)
log.plot(log=True)


# In[9]:

for i in range(3):
    ix = np.random.randint(len(val_ds))
    im,_ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1,2,figsize=(3,3)) 
    show(im[0], ax=ax[0], title='input')
    show(_im[0], ax=ax[1], title='prediction')
    plt.tight_layout()
    
    # Save instead of showing
    plt.savefig(f'./figs/simple_autoencoder_{i+1}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory
    


# In[10]: UMAP Projection Comparison: Original Data vs Autoencoder vs PCA

def extract_raw_data(dataloader, max_samples=2000):
    """Extract raw pixel data and labels from the validation set"""
    raw_data = []
    labels = []
    
    print("Extracting raw pixel data...")
    with torch.no_grad():
        samples_collected = 0
        for data, label in dataloader:
            if samples_collected >= max_samples:
                break
                
            # Flatten the images to vectors
            raw_pixels = data.view(data.size(0), -1).cpu().numpy()
            raw_data.append(raw_pixels)
            labels.append(label.numpy())
            
            samples_collected += len(data)
    
    raw_data = np.vstack(raw_data)[:max_samples]
    labels = np.hstack(labels)[:max_samples]
    
    print(f"Extracted {len(raw_data)} raw samples with dimension {raw_data.shape[1]}")
    return raw_data, labels

def extract_latent_features(model, dataloader, max_samples=2000):
    """Extract latent features and labels from the validation set"""
    model.eval()
    latent_features = []
    labels = []
    
    print("Extracting latent features...")
    with torch.no_grad():
        samples_collected = 0
        for data, label in dataloader:
            if samples_collected >= max_samples:
                break
                
            # Extract latent features
            latent = model.encode(data)
            latent_features.append(latent.cpu().numpy())
            labels.append(label.numpy())
            
            samples_collected += len(data)
    
    latent_features = np.vstack(latent_features)[:max_samples]
    labels = np.hstack(labels)[:max_samples]
    
    print(f"Extracted {len(latent_features)} latent features with dimension {latent_features.shape[1]}")
    return latent_features, labels

def apply_pca_reduction(data, n_components=50):
    """Apply PCA to reduce dimensionality"""
    from sklearn.decomposition import PCA
    
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
        verbose=False  # Reduced verbosity for multiple runs
    )
    
    embedding = reducer.fit_transform(features_scaled)
    
    print(f"UMAP projection completed for {method_name}. Embedding shape: {embedding.shape}")
    return embedding

def plot_umap_comparison(embeddings_dict, labels, latent_dim, save_path='./figs/simple_autoencoder_umap_comparison.png'):
    """Plot and save comparison of three UMAP projections"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['Original Data (784D)', f'Autoencoder Latent ({latent_dim}D)', f'PCA Reduced (50D)']
    
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
    fig.suptitle('UMAP Projections Comparison: Dimensionality Reduction Methods', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"UMAP comparison plot saved as: {save_path}")
    
    # Show the plot
    plt.show()
    
    return fig

def calculate_metrics(embedding, labels, method_name):
    """Calculate clustering quality metrics"""
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    
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

# Main Analysis
print("\n" + "="*60)
print("UMAP PROJECTIONS COMPARISON: ORIGINAL vs AUTOENCODER vs PCA")
print("="*60)

# Extract all data types
raw_data, labels = extract_raw_data(val_dl, max_samples=2000)
latent_features, labels = extract_latent_features(model, val_dl, max_samples=2000)

# Apply PCA to raw data
pca_features = apply_pca_reduction(raw_data, n_components=50)

# Apply UMAP to all three representations
print("\n" + "="*40)
print("APPLYING UMAP PROJECTIONS")
print("="*40)

embeddings = {}
embeddings['Original Data (784D)'] = apply_umap_projection(raw_data, 'Original Data')
embeddings[f'Autoencoder Latent ({latent_dim}D)'] = apply_umap_projection(latent_features, 'Autoencoder Latent')
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

print(f"{'Method':<30} {'Silhouette':<12} {'ARI':<8}")
print("-" * 50)
for method, metric in metrics.items():
    print(f"{method:<30} {metric['silhouette']:<12.3f} {metric['ari']:<8.3f}")

# Find best method for each metric
best_silhouette = max(metrics.keys(), key=lambda x: metrics[x]['silhouette'])
best_ari = max(metrics.keys(), key=lambda x: metrics[x]['ari'])

print(f"\nBest cluster separation (Silhouette): {best_silhouette}")
print(f"Best label agreement (ARI): {best_ari}")

print("\n" + "="*40)
print("ANALYSIS INSIGHTS")
print("="*40)
print("• Original Data (784D): Raw pixel intensities, high-dimensional")
print(f"• Autoencoder Latent ({latent_dim}D): Learned compressed representation")
print("• PCA Reduced (64D): Linear dimensionality reduction preserving variance")
print("\nHigher metrics indicate better separation of digit classes.")
print("The autoencoder should ideally show better clustering than PCA")
print("if it has learned meaningful non-linear representations.")

print(f"\nVisualization saved as: ./figs/simple_autoencoder_umap_comparison.png")
