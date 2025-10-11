#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PacktPublishing/Modern-Computer-Vision-with-PyTorch-2E/blob/main/Chapter11/conv_auto_encoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import torch
from torch import nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch_snippets import *
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from torch_snippets.torch_loader import Report
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.to(device))
])

trn_ds = MNIST('./data/', transform=img_transform, train=True, download=True)
val_ds = MNIST('./data/', transform=img_transform, train=False, download=True)

batch_size = 128
trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


# In[2]:


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1), nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode_only(self, x):
        """Extract latent features only"""
        return self.encoder(x)

model = ConvAutoEncoder().to(device)
from torchsummary import summary
summary(model, torch.zeros(2,1,28,28));


# In[3]:


def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss


# In[4]:


@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss


# In[5]:


model = ConvAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)


# In[6]:


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


# In[7]:


log.plot_epochs(log=True)


# In[8]:


for i in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1, 2, figsize=(3,3))
    show(im[0], ax=ax[0], title='input')
    show(_im[0], ax=ax[1], title='prediction')
    plt.tight_layout()    
    # Save instead of showing
    plt.savefig(f'./figs/conv_autoencoder_{i+1}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory


# In[9]: UMAP Projection Comparison


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
            
            # Raw pixel data
            raw_pixels = data.view(data.size(0), -1).cpu().numpy()
            raw_data.append(raw_pixels)
            
            # Latent features from encoder
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

def apply_pca_reduction(data, n_components):
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

def plot_umap_comparison(embeddings_dict, labels, latent_dim, save_path='./figs/conv_autoencoder_umap_comparison.png'):
    """Plot and save comparison of three UMAP projections"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['Original Data (784D)', f'Conv Encoder Latent ({latent_dim}D)', f'PCA Reduced ({latent_dim}D)']
    
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
    fig.suptitle('UMAP Projections Comparison: Conv Autoencoder vs PCA', 
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
print("UMAP PROJECTIONS COMPARISON: CONV AUTOENCODER ANALYSIS")
print("="*60)

# Extract all data types
raw_data, latent_features, labels = extract_data_for_comparison(model, val_dl, max_samples=2000)

# Determine latent dimension
latent_dim = latent_features.shape[1]
print(f"Encoder latent dimension: {latent_dim}")

# Apply PCA to raw data (same dimension as latent space)
pca_features = apply_pca_reduction(raw_data, n_components=latent_dim)

# Apply UMAP to all three representations
print("\n" + "="*40)
print("APPLYING UMAP PROJECTIONS")
print("="*40)

embeddings = {}
embeddings['Original Data (784D)'] = apply_umap_projection(raw_data, 'Original Data')
embeddings[f'Conv Encoder Latent ({latent_dim}D)'] = apply_umap_projection(latent_features, 'Conv Encoder Latent')
embeddings[f'PCA Reduced ({latent_dim}D)'] = apply_umap_projection(pca_features, 'PCA Reduced')

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

print(f"{'Method':<35} {'Silhouette':<12} {'ARI':<8}")
print("-" * 55)
for method, metric in metrics.items():
    print(f"{method:<35} {metric['silhouette']:<12.3f} {metric['ari']:<8.3f}")

# Find best method for each metric
best_silhouette = max(metrics.keys(), key=lambda x: metrics[x]['silhouette'])
best_ari = max(metrics.keys(), key=lambda x: metrics[x]['ari'])

print(f"\nBest cluster separation (Silhouette): {best_silhouette}")
print(f"Best label agreement (ARI): {best_ari}")

print("\n" + "="*40)
print("ANALYSIS INSIGHTS")
print("="*40)
print("• Original Data (784D): Raw pixel intensities, high-dimensional")
print(f"• Conv Encoder Latent ({latent_dim}D): Spatial hierarchical features learned by conv layers")
print(f"• PCA Reduced ({latent_dim}D): Linear dimensionality reduction preserving variance")
print("\nHigher metrics indicate better separation of digit classes.")
print("The convolutional encoder should capture spatial patterns better than PCA")
print("and show superior clustering due to hierarchical feature learning.")

print(f"\nVisualization saved as: ./figs/conv_autoencoder_umap_comparison.png")


# In[10]: Original t-SNE comparison (from original code)


latent_vectors = []
classes = []


# In[11]:


for im,clss in val_dl:
    latent_vectors.append(model.encoder(im).view(len(im),-1))
    classes.extend(clss)


# In[12]:


latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy()


# In[13]:


from sklearn.manifold import TSNE
tsne = TSNE(2)


# In[14]:


clustered = tsne.fit_transform(latent_vectors)


# In[15]:


fig = plt.figure(figsize=(12,10))
cmap = plt.get_cmap('Spectral', 10)
plt.scatter(*zip(*clustered), c=classes, cmap=cmap)
plt.colorbar(drawedges=True)
plt.title('t-SNE Projection of Conv Autoencoder Latent Space', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('./figs/conv_autoencoder_tsne.png', dpi=300, bbox_inches='tight')
plt.show()


# In[16]: Random generation (from original code)


latent_vectors = []
classes = []
for im,clss in val_dl:
    latent_vectors.append(model.encoder(im))
    classes.extend(clss)
latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy().reshape(10000, -1)


# In[17]:


rand_vectors = []
for col in latent_vectors.transpose(1,0):
    mu, sigma = col.mean(), col.std()
    rand_vectors.append(sigma*torch.randn(1,100) + mu)


# In[18]:


rand_vectors = torch.cat(rand_vectors).transpose(1,0).to(device)
fig, ax = plt.subplots(10,10,figsize=(7,7)); ax = iter(ax.flat)
for p in rand_vectors:
    img = model.decoder(p.reshape(1,64,2,2)).view(28,28)
    show(img, ax=next(ax))

plt.suptitle('Random Generated Images from Latent Space', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./figs/conv_autoencoder_random_generation.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:
