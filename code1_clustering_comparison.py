import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

# Clustering algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize

# Evaluation metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Visualization
import umap
import seaborn as sns
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# DINO Model Classes (copy from your working implementation)
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
    """Improved CNN Encoder - copy from your DINO implementation"""
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            self._make_layer(64, 64, stride=1),
            self._make_layer(64, 64, stride=1),
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 128, stride=1),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 256, stride=1),
            self._make_layer(256, 512, stride=2),
            self._make_layer(512, 512, stride=1),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.feature_dim = 512
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
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
    """DINO projection head"""
    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=2048, nlayers=3):
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
        x = F.normalize(x, dim=-1, p=2)
        return x

class FixedMultiCropDINO(nn.Module):
    """Fixed Multi-Crop DINO - copy from your implementation"""
    def __init__(self, input_channels=3, out_dim=2048):
        super().__init__()
        
        self.student_encoder = ImprovedCNNEncoder(input_channels)
        self.teacher_encoder = ImprovedCNNEncoder(input_channels)
        
        feature_dim = self.student_encoder.feature_dim
        self.student_head = DINOProjectionHead(feature_dim, 1024, out_dim)
        self.teacher_head = DINOProjectionHead(feature_dim, 1024, out_dim)
        
        self._initialize_teacher()
        
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
            
        self.register_buffer('center', torch.zeros(1, out_dim))
        
    def _initialize_teacher(self):
        for student_param, teacher_param in zip(self.student_encoder.parameters(), 
                                              self.teacher_encoder.parameters()):
            teacher_param.data.copy_(student_param.data)
        
        for student_param, teacher_param in zip(self.student_head.parameters(), 
                                              self.teacher_head.parameters()):
            teacher_param.data.copy_(student_param.data)
    
    def encode_images(self, imgs):
        """Extract features for evaluation (use student encoder)"""
        with torch.no_grad():
            features = self.student_encoder(imgs)
        return features

def load_dino_model(model_path, device):
    """Load trained DINO model"""
    print(f"Loading DINO model from: {model_path}")
    
    # Initialize model with same parameters as training
    model = FixedMultiCropDINO(input_channels=3, out_dim=2048).to(device)
    
    # Load trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("DINO model loaded successfully")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    return model

def extract_features_from_dataset(model, data_dir, device, batch_size=32):
    """Extract features from entire dataset using trained DINO model"""
    print("Extracting features from entire dataset...")
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    all_features = []
    all_labels = []
    all_indices = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, indices) in enumerate(dataloader):
            images = images.to(device)
            
            # Extract features using student encoder
            features = model.encode_images(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_indices.extend(indices.numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx}/{len(dataloader)}")
    
    features = np.vstack(all_features)
    labels = np.array(all_labels)
    indices = np.array(all_indices)
    
    print(f"Extracted features: {features.shape}")
    print(f"Number of samples: {len(labels)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    return features, labels, indices

class ClusteringComparison:
    """Comprehensive clustering algorithm comparison"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.n_clusters = len(np.unique(labels))
        self.n_samples = len(labels)
        
        print(f"Clustering comparison setup:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Features: {features.shape[1]} (full DINO features)")
        print(f"  True clusters: {self.n_clusters}")
        
        # Standardize features for Euclidean distance-based clustering
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(features)
        
        # Normalize for cosine similarity (only for DBSCAN)
        self.features_normalized = normalize(self.features_scaled, norm='l2')
        
        # Use standardized features for most algorithms (Euclidean distance)
        self.features_euclidean = self.features_scaled
        print(f"  Using standardized {self.features_euclidean.shape[1]}D DINO features (Euclidean distance)")
        print(f"  Using L2-normalized features only for DBSCAN and GMM (cosine similarity)")
        
        self.results = {}
    
    def run_kmeans(self):
        """Centroid-based: K-means clustering with Euclidean distance"""
        print("\n1. Running K-means clustering...")
        
        # Use standardized features for Euclidean distance
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(self.features_euclidean)
        
        # Compute metrics
        ari = adjusted_rand_score(self.labels, pred_labels)
        nmi = normalized_mutual_info_score(self.labels, pred_labels)
        silhouette = silhouette_score(self.features_euclidean, pred_labels, metric='euclidean')
        
        self.results['K-means'] = {
            'category': 'Centroid-based',
            'pred_labels': pred_labels,
            'ari': ari,
            'nmi': nmi,
            'silhouette': silhouette,
            'algorithm': kmeans
        }
        
        print(f"  K-means - ARI: {ari:.3f}, NMI: {nmi:.3f}, Silhouette: {silhouette:.3f}")
    

            
    
    def run_dbscan(self):
        """Improved DBSCAN clustering with cosine distance (ONLY algorithm using cosine)"""
        print("\n2. Running DBSCAN clustering...")
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Adaptive subset selection
            subset_size = min(1200, max(500, self.n_samples // 2))
            if self.n_samples > subset_size:
                print(f"  Using subset of {subset_size} samples for parameter estimation")
                subset_idx = np.random.choice(self.n_samples, subset_size, replace=False)
                features_subset = self.features_normalized[subset_idx]  # Use normalized features for cosine
            else:
                features_subset = self.features_normalized
                subset_idx = np.arange(self.n_samples)
            
            # Multi-strategy eps estimation for cosine distance
            eps_candidates = []
            
            # Strategy 1: k-distance plot with cosine distance
            k_values = [4, 6, 8, 10, 12, 15]
            print(f"  Strategy 1: k-distance analysis with cosine metric")
            
            for k in k_values:
                try:
                    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
                    nn.fit(features_subset)
                    distances, _ = nn.kneighbors(features_subset)
                    k_distances = np.sort(distances[:, k-1])
                    
                    # Multiple percentile-based eps values
                    for percentile in [25, 50, 75, 85, 90]:
                        eps = np.percentile(k_distances, percentile)
                        eps_candidates.append(('k-dist', f'k={k},p={percentile}', eps))
                        
                    # Knee detection in k-distance plot
                    try:
                        # Simple knee detection using second derivative
                        if len(k_distances) > 50:
                            n_points = min(200, len(k_distances))
                            indices = np.linspace(0, len(k_distances)-1, n_points, dtype=int)
                            y = k_distances[indices]
                            x = np.arange(len(y))
                            
                            # Compute second derivative
                            second_derivative = np.gradient(np.gradient(y))
                            knee_idx = np.argmax(second_derivative)
                            knee_eps = y[knee_idx]
                            
                            eps_candidates.append(('knee', f'k={k}', knee_eps))
                            
                    except:
                        pass
                        
                except Exception as e:
                    print(f"    k={k} failed: {str(e)[:30]}...")
                    continue
            
            # Strategy 2: Local density-based eps estimation
            print(f"  Strategy 2: Local density analysis")
            try:
                nn = NearestNeighbors(n_neighbors=20, metric='cosine')
                nn.fit(features_subset)
                distances, _ = nn.kneighbors(features_subset)
                
                # Compute local densities
                local_densities = 1.0 / (np.mean(distances, axis=1) + 1e-8)
                density_sorted_idx = np.argsort(local_densities)[::-1]
                
                # Use different density thresholds
                for top_percent in [10, 25, 50]:
                    n_top = int(len(features_subset) * top_percent / 100)
                    top_dense_points = density_sorted_idx[:n_top]
                    
                    # Eps as distance to maintain connectivity among dense points
                    if len(top_dense_points) > 1:
                        nn_dense = NearestNeighbors(n_neighbors=min(5, len(top_dense_points)), metric='cosine')
                        nn_dense.fit(features_subset[top_dense_points])
                        dense_distances, _ = nn_dense.kneighbors(features_subset[top_dense_points])
                        eps = np.median(dense_distances[:, -1])
                        eps_candidates.append(('density', f'top{top_percent}%', eps))
                        
            except Exception as e:
                print(f"    Density analysis failed: {str(e)[:30]}...")
            
            # Strategy 3: Target cluster size based eps
            print(f"  Strategy 3: Target cluster size analysis")
            try:
                expected_cluster_size = self.n_samples / self.n_clusters
                for target_ratio in [0.5, 0.8, 1.0, 1.5, 2.0]:
                    target_neighbors = int(expected_cluster_size * target_ratio)
                    k_target = min(target_neighbors, len(features_subset) // 2)
                    
                    if k_target >= 2:
                        nn = NearestNeighbors(n_neighbors=k_target, metric='cosine')
                        nn.fit(features_subset)
                        distances, _ = nn.kneighbors(features_subset)
                        eps = np.median(distances[:, -1])
                        eps_candidates.append(('target', f'ratio={target_ratio}', eps))
                        
            except Exception as e:
                print(f"    Target size analysis failed: {str(e)[:30]}...")
            
            # Remove duplicates and filter reasonable eps values
            unique_eps = {}
            for strategy, desc, eps in eps_candidates:
                if 0.001 < eps < 0.8:  # Reasonable range for cosine distance
                    key = round(eps, 4)
                    if key not in unique_eps:
                        unique_eps[key] = (strategy, desc, eps)
            
            eps_candidates = list(unique_eps.values())
            eps_candidates.sort(key=lambda x: x[2])  # Sort by eps value
            
            # Adaptive min_samples based on dataset characteristics
            base_min_samples = max(2, int(np.log2(len(features_subset))))
            min_samples_candidates = [
                base_min_samples,
                base_min_samples + 2,
                max(3, base_min_samples - 1),
                min(base_min_samples * 2, 20),
                max(2, int(expected_cluster_size * 0.1)),
                max(2, int(expected_cluster_size * 0.05))
            ]
            min_samples_candidates = sorted(list(set(min_samples_candidates)))
            
            print(f"  Testing {len(eps_candidates)} eps values with {len(min_samples_candidates)} min_samples")
            print(f"  Min_samples candidates: {min_samples_candidates}")
            
            best_result = None
            best_score = -1
            
            for strategy, desc, eps in eps_candidates:
                for min_samples in min_samples_candidates:
                    try:
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                        pred_labels_subset = dbscan.fit_predict(features_subset)
                        
                        # Count clusters and noise
                        unique_labels = set(pred_labels_subset)
                        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                        n_noise = list(pred_labels_subset).count(-1)
                        noise_ratio = n_noise / len(pred_labels_subset)
                        
                        # Enhanced filtering criteria
                        min_clusters = max(2, self.n_clusters // 2)
                        max_clusters = min(self.n_clusters * 4, self.n_samples // 5)
                        max_noise_ratio = 0.6
                        
                        if (n_clusters_found < min_clusters or 
                            n_clusters_found > max_clusters or 
                            noise_ratio > max_noise_ratio):
                            continue
                        
                        print(f"    {strategy}({desc}) eps={eps:.4f}, min_samples={min_samples} → "
                              f"{n_clusters_found} clusters, {noise_ratio:.1%} noise")
                        
                        # Extend to full dataset
                        if self.n_samples > subset_size:
                            # Sophisticated extension for cosine metric
                            if hasattr(dbscan, 'core_sample_indices_') and len(dbscan.core_sample_indices_) > 0:
                                core_samples = features_subset[dbscan.core_sample_indices_]
                                core_labels = pred_labels_subset[dbscan.core_sample_indices_]
                                
                                # Create mapping from core samples to full dataset
                                nn_full = NearestNeighbors(n_neighbors=1, metric='cosine')
                                nn_full.fit(core_samples)
                                distances, indices = nn_full.kneighbors(self.features_normalized)
                                
                                pred_labels = np.full(self.n_samples, -1)
                                
                                # Assign labels based on distance to core samples
                                for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
                                    if dist <= eps:
                                        pred_labels[i] = core_labels[idx]
                                    else:
                                        # Secondary assignment for border points
                                        nn_k = NearestNeighbors(n_neighbors=min(3, len(core_samples)), metric='cosine')
                                        nn_k.fit(core_samples)
                                        k_distances, k_indices = nn_k.kneighbors([self.features_normalized[i]])
                                        
                                        # If majority of k nearest core samples are within 1.5*eps, assign to mode
                                        close_mask = k_distances[0] <= 1.5 * eps
                                        if np.sum(close_mask) >= 2:
                                            close_labels = core_labels[k_indices[0][close_mask]]
                                            pred_labels[i] = np.bincount(close_labels).argmax()
                            else:
                                pred_labels = np.full(self.n_samples, -1)
                        else:
                            pred_labels = pred_labels_subset
                        
                        # Compute metrics for non-noise points
                        non_noise_mask = pred_labels != -1
                        if np.sum(non_noise_mask) > 10 and len(np.unique(pred_labels[non_noise_mask])) > 1:
                            ari = adjusted_rand_score(self.labels[non_noise_mask], pred_labels[non_noise_mask])
                            nmi = normalized_mutual_info_score(self.labels[non_noise_mask], pred_labels[non_noise_mask])
                            
                            try:
                                silhouette = silhouette_score(self.features_normalized[non_noise_mask], 
                                                            pred_labels[non_noise_mask], metric='cosine')
                            except:
                                silhouette = 0.0
                            
                            # Enhanced composite scoring
                            final_noise_ratio = np.sum(pred_labels == -1) / len(pred_labels)
                            final_n_clusters = len(np.unique(pred_labels[pred_labels != -1]))
                            
                            # Penalties and bonuses
                            noise_penalty = final_noise_ratio * 0.3
                            cluster_penalty = abs(final_n_clusters - self.n_clusters) / self.n_clusters * 0.2
                            
                            # Bonus for balanced cluster sizes
                            if final_n_clusters > 1:
                                cluster_sizes = np.bincount(pred_labels[pred_labels != -1])
                                size_balance = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
                                balance_bonus = max(0, size_balance * 0.1)
                            else:
                                balance_bonus = 0
                            
                            composite_score = ari * (1 - noise_penalty - cluster_penalty + balance_bonus)
                            
                            if composite_score > best_score:
                                best_score = composite_score
                                best_result = {
                                    'pred_labels': pred_labels,
                                    'ari': ari,
                                    'nmi': nmi,
                                    'silhouette': silhouette,
                                    'eps': eps,
                                    'min_samples': min_samples,
                                    'strategy': f"{strategy}({desc})",
                                    'n_clusters_found': final_n_clusters,
                                    'n_noise': np.sum(pred_labels == -1),
                                    'noise_ratio': final_noise_ratio,
                                    'composite_score': composite_score
                                }
                    
                    except Exception as e:
                        continue
            
            if best_result is not None:
                self.results['DBSCAN'] = {
                    'category': 'Density-based',
                    'pred_labels': best_result['pred_labels'],
                    'ari': best_result['ari'],
                    'nmi': best_result['nmi'],
                    'silhouette': best_result['silhouette'],
                    'algorithm': f"DBSCAN(eps={best_result['eps']:.4f}, min_samples={best_result['min_samples']}, metric='cosine')",
                    'eps': best_result['eps'],
                    'min_samples': best_result['min_samples'],
                    'strategy': best_result['strategy'],
                    'n_clusters_found': best_result['n_clusters_found'],
                    'n_noise': best_result['n_noise'],
                    'noise_ratio': best_result['noise_ratio']
                }
                
                print(f"  DBSCAN - Best: {best_result['strategy']}")
                print(f"  DBSCAN - eps: {best_result['eps']:.4f}, min_samples: {best_result['min_samples']}")
                print(f"  DBSCAN - Clusters: {best_result['n_clusters_found']}, Noise: {best_result['noise_ratio']:.1%}")
                print(f"  DBSCAN - ARI: {best_result['ari']:.3f}, NMI: {best_result['nmi']:.3f}, Silhouette: {best_result['silhouette']:.3f}")
            else:
                print("  DBSCAN - Could not find suitable parameters")
                self.results['DBSCAN'] = {
                    'category': 'Density-based',
                    'pred_labels': np.zeros(self.n_samples),
                    'ari': 0.0,
                    'nmi': 0.0,
                    'silhouette': 0.0,
                    'algorithm': None,
                    'error': 'No suitable parameters found'
                }
                
        except Exception as e:
            print(f"  DBSCAN completely failed: {e}")
            self.results['DBSCAN'] = {
                'category': 'Density-based',
                'pred_labels': np.zeros(self.n_samples),
                'ari': 0.0,
                'nmi': 0.0,
                'silhouette': 0.0,
                'algorithm': None,
                'error': str(e)
            }
    
    def run_agglomerative(self):
        """Hierarchical: Agglomerative clustering with Euclidean distance"""
        print("\n3. Running Agglomerative clustering...")
        
        try:
            # Try different linkage methods with Euclidean distance
            linkage_methods = ['single', 'complete', 'average', 'ward']  # ward works with Euclidean
            best_result = None
            best_score = -1
            
            for linkage in linkage_methods:
                try:
                    if linkage == 'ward':
                        # Ward linkage requires Euclidean distance
                        agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=linkage)
                    else:
                        agg = AgglomerativeClustering(n_clusters=self.n_clusters, 
                                                    metric='euclidean', linkage=linkage)
                    pred_labels = agg.fit_predict(self.features_euclidean)
                    
                    # Compute metrics
                    ari = adjusted_rand_score(self.labels, pred_labels)
                    nmi = normalized_mutual_info_score(self.labels, pred_labels)
                    silhouette = silhouette_score(self.features_euclidean, pred_labels, metric='euclidean')
                    
                    print(f"    {linkage} linkage - ARI: {ari:.3f}")
                    
                    if ari > best_score:
                        best_score = ari
                        best_result = {
                            'pred_labels': pred_labels,
                            'ari': ari,
                            'nmi': nmi,
                            'silhouette': silhouette,
                            'algorithm': agg,
                            'linkage': linkage
                        }
                        
                except Exception as e:
                    print(f"    {linkage} linkage failed: {str(e)[:50]}...")
                    continue
            
            if best_result is not None:
                self.results['Agglomerative'] = {
                    'category': 'Hierarchical',
                    'pred_labels': best_result['pred_labels'],
                    'ari': best_result['ari'],
                    'nmi': best_result['nmi'],
                    'silhouette': best_result['silhouette'],
                    'algorithm': best_result['algorithm'],
                    'linkage': best_result['linkage']
                }
                
                print(f"  Agglomerative - Best linkage: {best_result['linkage']}")
                print(f"  Agglomerative - ARI: {best_result['ari']:.3f}, NMI: {best_result['nmi']:.3f}, Silhouette: {best_result['silhouette']:.3f}")
            else:
                raise Exception("All linkage methods failed")
                
        except Exception as e:
            print(f"  Agglomerative failed: {e}")
            self.results['Agglomerative'] = {
                'category': 'Hierarchical',
                'pred_labels': np.zeros(self.n_samples),
                'ari': 0.0,
                'nmi': 0.0,
                'silhouette': 0.0,
                'algorithm': None,
                'error': str(e)
            }
    
    def run_gmm(self):
        """Probabilistic: Gaussian Mixture Model with normalized features (Cosine similarity)"""
        print("\n4. Running Gaussian Mixture Model...")
        
        try:
            # GMM works better with normalized features for cosine similarity
            configs = [
                {'covariance_type': 'full', 'reg_covar': 1e-6},
                {'covariance_type': 'diag', 'reg_covar': 1e-6},
                {'covariance_type': 'tied', 'reg_covar': 1e-4},
                {'covariance_type': 'spherical', 'reg_covar': 1e-4}
            ]
            
            best_result = None
            best_score = -np.inf
            
            for config in configs:
                try:
                    gmm = GaussianMixture(
                        n_components=self.n_clusters, 
                        random_state=42,
                        max_iter=100,
                        **config
                    )
                    gmm.fit(self.features_normalized)  # Use normalized features for cosine similarity
                    pred_labels = gmm.predict(self.features_normalized)
                    
                    # Compute metrics
                    ari = adjusted_rand_score(self.labels, pred_labels)
                    nmi = normalized_mutual_info_score(self.labels, pred_labels)
                    silhouette = silhouette_score(self.features_normalized, pred_labels, metric='cosine')
                    
                    if ari > best_score:
                        best_score = ari
                        best_result = {
                            'pred_labels': pred_labels,
                            'ari': ari,
                            'nmi': nmi,
                            'silhouette': silhouette,
                            'algorithm': gmm,
                            'bic': gmm.bic(self.features_normalized),
                            'aic': gmm.aic(self.features_normalized),
                            'config': config
                        }
                        
                    print(f"    {config['covariance_type']} covariance - ARI: {ari:.3f}")
                    
                except Exception as inner_e:
                    print(f"    {config['covariance_type']} covariance failed: {str(inner_e)[:50]}...")
                    continue
            
            if best_result is not None:
                self.results['GMM'] = {
                    'category': 'Probabilistic',
                    'pred_labels': best_result['pred_labels'],
                    'ari': best_result['ari'],
                    'nmi': best_result['nmi'],
                    'silhouette': best_result['silhouette'],
                    'algorithm': best_result['algorithm'],
                    'bic': best_result['bic'],
                    'aic': best_result['aic'],
                    'best_config': best_result['config']
                }
                
                print(f"  GMM - Best config: {best_result['config']['covariance_type']}")
                print(f"  GMM - ARI: {best_result['ari']:.3f}, NMI: {best_result['nmi']:.3f}, Silhouette: {best_result['silhouette']:.3f}")
                print(f"  GMM - BIC: {best_result['bic']:.1f}, AIC: {best_result['aic']:.1f}")
            else:
                raise Exception("All GMM configurations failed")
                
        except Exception as e:
            print(f"  GMM completely failed: {e}")
            self.results['GMM'] = {
                'category': 'Probabilistic',
                'pred_labels': np.zeros(self.n_samples),
                'ari': 0.0,
                'nmi': 0.0,
                'silhouette': 0.0,
                'algorithm': None,
                'error': str(e)
            }
    
    def run_spectral(self):
        """Graph-based: Spectral clustering with Euclidean affinity"""
        print("\n5. Running Spectral clustering...")
        
        try:
            # Use subset for large datasets (Spectral clustering is memory intensive)
            if self.n_samples > 1500:
                print("  Using subset for Spectral clustering (memory efficiency)")
                subset_idx = np.random.choice(self.n_samples, 1500, replace=False)
                features_subset = self.features_euclidean[subset_idx]
                
                # Try different affinity methods with Euclidean distance
                affinity_methods = [
                    ('euclidean', {}),
                    ('nearest_neighbors', {'n_neighbors': 10}),
                    ('nearest_neighbors', {'n_neighbors': 20})
                ]
                
                best_result = None
                best_score = -1
                
                for affinity, params in affinity_methods:
                    try:
                        spectral = SpectralClustering(n_clusters=self.n_clusters, random_state=42, 
                                                    affinity=affinity, **params)
                        pred_labels_subset = spectral.fit_predict(features_subset)
                        
                        # Extend to full dataset using nearest neighbors
                        from sklearn.neighbors import KNeighborsClassifier
                        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
                        knn.fit(features_subset, pred_labels_subset)
                        pred_labels = knn.predict(self.features_euclidean)
                        
                        # Compute metrics
                        ari = adjusted_rand_score(self.labels, pred_labels)
                        nmi = normalized_mutual_info_score(self.labels, pred_labels)
                        silhouette = silhouette_score(self.features_euclidean, pred_labels, metric='euclidean')
                        
                        print(f"    {affinity} affinity - ARI: {ari:.3f}")
                        
                        if ari > best_score:
                            best_score = ari
                            best_result = {
                                'pred_labels': pred_labels,
                                'ari': ari,
                                'nmi': nmi,
                                'silhouette': silhouette,
                                'algorithm': spectral,
                                'affinity': affinity,
                                'params': params
                            }
                            
                    except Exception as e:
                        print(f"    {affinity} affinity failed: {str(e)[:50]}...")
                        continue
                
                if best_result is None:
                    raise Exception("All affinity methods failed for subset")
                    
                pred_labels = best_result['pred_labels']
                ari = best_result['ari']
                nmi = best_result['nmi']
                silhouette = best_result['silhouette']
                spectral = best_result['algorithm']
                
            else:
                # Use Euclidean affinity for full dataset
                spectral = SpectralClustering(n_clusters=self.n_clusters, random_state=42,
                                            affinity='nearest_neighbors')
                pred_labels = spectral.fit_predict(self.features_euclidean)
                
                # Compute metrics
                ari = adjusted_rand_score(self.labels, pred_labels)
                nmi = normalized_mutual_info_score(self.labels, pred_labels)
                silhouette = silhouette_score(self.features_euclidean, pred_labels, metric='euclidean')
            
            self.results['Spectral'] = {
                'category': 'Graph-based',
                'pred_labels': pred_labels,
                'ari': ari,
                'nmi': nmi,
                'silhouette': silhouette,
                'algorithm': spectral
            }
            
            print(f"  Spectral - ARI: {ari:.3f}, NMI: {nmi:.3f}, Silhouette: {silhouette:.3f}")
            
        except Exception as e:
            print(f"  Spectral clustering failed: {e}")
            self.results['Spectral'] = {
                'category': 'Graph-based',
                'pred_labels': np.zeros(self.n_samples),
                'ari': 0.0,
                'nmi': 0.0,
                'silhouette': 0.0,
                'algorithm': None,
                'error': str(e)
            }
    
    def run_all_algorithms(self):
        """Run all clustering algorithms"""
        print("="*60)
        print("COMPREHENSIVE CLUSTERING COMPARISON (EUCLIDEAN DISTANCE)")
        print("(DBSCAN and GMM use cosine similarity)")
        print("="*60)
        
        self.run_kmeans()
        self.run_dbscan()
        self.run_agglomerative()
        self.run_gmm()
        self.run_spectral()
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60)
        
        return self.results
    
    def create_results_summary(self):
        """Create a comprehensive results summary DataFrame"""
        print("\nCreating results summary...")
        
        summary_data = []
        
        for algo_name, result in self.results.items():
            row = {
                'Algorithm': algo_name,
                'Category': result.get('category', 'Unknown'),
                'ARI': result['ari'],
                'NMI': result['nmi'],
                'Silhouette': result['silhouette'],
                'Status': 'Success' if 'error' not in result else 'Failed'
            }
            
            # Add algorithm-specific details
            if algo_name == 'DBSCAN' and 'n_clusters_found' in result:
                row['Clusters_Found'] = result['n_clusters_found']
                row['Noise_Ratio'] = result.get('noise_ratio', 0)
                row['Parameters'] = f"eps={result.get('eps', 'N/A'):.4f}, min_samples={result.get('min_samples', 'N/A')}"
            elif algo_name == 'Mean Shift' and 'n_clusters_found' in result:
                row['Clusters_Found'] = result['n_clusters_found']
                row['Parameters'] = f"bandwidth={result.get('bandwidth', 'N/A'):.4f}"
            elif algo_name == 'Agglomerative' and 'linkage' in result:
                row['Clusters_Found'] = self.n_clusters
                row['Parameters'] = f"linkage={result.get('linkage', 'N/A')}"
            elif algo_name == 'GMM' and 'best_config' in result:
                row['Clusters_Found'] = self.n_clusters
                row['Parameters'] = f"covariance={result.get('best_config', {}).get('covariance_type', 'N/A')}, cosine similarity"
            elif algo_name == 'Spectral':
                row['Clusters_Found'] = self.n_clusters
                row['Parameters'] = "affinity=euclidean"
            elif algo_name == 'K-means':
                row['Clusters_Found'] = self.n_clusters
                row['Parameters'] = "Euclidean distance (standardized features)"
            else:
                row['Clusters_Found'] = 'N/A'
                row['Parameters'] = 'N/A'
            
            if 'error' in result:
                row['Error'] = result['error']
            
            summary_data.append(row)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by ARI score (descending)
        summary_df = summary_df.sort_values('ARI', ascending=False)
        
        # Display summary
        print("\n" + "="*80)
        print("CLUSTERING RESULTS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False, float_format='%.3f'))
        
        # Best performing algorithm
        best_algo = summary_df.iloc[0]
        print(f"\nBest performing algorithm: {best_algo['Algorithm']}")
        print(f"  Category: {best_algo['Category']}")
        print(f"  ARI: {best_algo['ARI']:.3f}")
        print(f"  NMI: {best_algo['NMI']:.3f}")
        print(f"  Silhouette: {best_algo['Silhouette']:.3f}")
        if best_algo['Parameters'] != 'N/A':
            print(f"  Parameters: {best_algo['Parameters']}")
        
        return summary_df
    
    def visualize_results(self, save_path='./figs/clustering_comparison.png'):
        """Create comprehensive visualization of results"""
        print("\nCreating visualization...")
        
        # Compute UMAP for visualization - using Euclidean distance for consistency
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, 
                               min_dist=0.1, metric='euclidean')
        features_umap = umap_reducer.fit_transform(self.features_euclidean)
        
        # Create subplot grid
        n_algorithms = len(self.results)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Plot true labels
        scatter = axes[0].scatter(features_umap[:, 0], features_umap[:, 1], 
                                c=self.labels, cmap='tab10', alpha=0.7, s=15)
        axes[0].set_title('True Labels', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        
        # Plot each algorithm's results
        for idx, (algo_name, result) in enumerate(self.results.items(), 1):
            if idx < len(axes):
                scatter = axes[idx].scatter(features_umap[:, 0], features_umap[:, 1], 
                                          c=result['pred_labels'], cmap='viridis', alpha=0.7, s=15)
                
                title = f"{algo_name}\nARI: {result['ari']:.3f}"
                axes[idx].set_title(title, fontweight='bold', fontsize=10)
                axes[idx].set_xlabel('UMAP 1')
                axes[idx].set_ylabel('UMAP 2')
        
        # Hide unused subplots
        for idx in range(len(self.results) + 1, len(axes)):
            axes[idx].axis('off')
        
        # Add results bar chart in last subplot
        if len(axes) > len(self.results) + 1:
            algorithms = list(self.results.keys())
            ari_scores = [self.results[algo]['ari'] for algo in algorithms]
            
            bars = axes[-1].bar(range(len(algorithms)), ari_scores, alpha=0.7)
            axes[-1].set_xticks(range(len(algorithms)))
            axes[-1].set_xticklabels(algorithms, rotation=45, ha='right')
            axes[-1].set_ylabel('ARI Score')
            axes[-1].set_title('ARI Comparison', fontweight='bold')
            axes[-1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, ari_scores):
                axes[-1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Clustering Algorithm Comparison on DINO Features (Euclidean Distance, except DBSCAN & GMM with Cosine)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        plt.show()
        
        return fig
    
def main():
    """Main function to run clustering comparison"""
    
    # Paths
    DATA_DIR = "../UnsupervisedFeatureLearningCNNs/data/corel"
    MODEL_PATH = "../UnsupervisedFeatureLearningCNNs/outputs/best_working_dino.pth"
    
    print("CLUSTERING ALGORITHM COMPARISON")
    print("="*50)
    print(f"Data directory: {DATA_DIR}")
    print(f"Model path: {MODEL_PATH}")
    
    # Load trained DINO model
    model = load_dino_model(MODEL_PATH, device)
    
    # Extract features from entire dataset
    features, labels, indices = extract_features_from_dataset(model, DATA_DIR, device)
    
    # Initialize clustering comparison
    clustering_comp = ClusteringComparison(features, labels)
    
    # Run all clustering algorithms
    results = clustering_comp.run_all_algorithms()
    
    # Create results summary
    summary_df = clustering_comp.create_results_summary()
    
    # Create visualization
    clustering_comp.visualize_results('./figs/clustering_comparison_results.png')
    
    # Save detailed results
    import pickle
    with open('clustering_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'features': features,
            'labels': labels,
            'summary': summary_df
        }, f)
    
    print("\nDetailed results saved to: clustering_results.pkl")
    print("Analysis complete!")
    
    return results, summary_df

if __name__ == "__main__":
    results, summary = main()
