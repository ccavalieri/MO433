#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering Evaluation
"""

import numpy as np
import pickle
import json
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse


CLASS_NAMES = {
    1: 'British Guards',
    2: 'Locomotives',
    3: 'Desserts',
    4: 'Salads',
    5: 'Snow',
    6: 'Sunset'
}

CLASS_COLORS = {
    1: '#e41a1c',  # Red
    2: '#377eb8',  # Blue
    3: '#4daf4a',  # Green
    4: '#984ea3',  # Purple
    5: '#ff7f00',  # Orange
    6: '#ffff33'   # Yellow
}


def load_features(feature_path):
    """Load features from pickle file"""
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    return data['features'], data['labels'], data['filenames']


def load_scenario_features(method_name, scenario, feature_paths):
    """Load features for a specific method and scenario"""
    print(f"\nLoading features: {method_name} - {scenario}")
    
    if scenario == 'Original':
        features, labels, filenames = load_features(feature_paths[method_name])
    
    elif scenario == '+LoRA':
        orig_features, orig_labels, orig_files = load_features(feature_paths[method_name])
        
        lora_dir = Path('/content/MO433/Trabalho Final/main/generated_images_corel')
        lora_images = []
        lora_labels = []
        
        for class_id in range(1, 7):
            class_name = list(lora_dir.glob(f'*'))[class_id-1].name
            class_dir = lora_dir / class_name
            class_images = sorted(list(class_dir.glob('*.png')))
            
            for img_file in class_images:
                if 'synthetic' in img_file.name:
                    lora_images.append(img_file.name)
                    lora_labels.append(class_id)
        
        features = np.concatenate([orig_features, orig_features[:len(lora_images)]])
        labels = np.concatenate([orig_labels, np.array(lora_labels)])
        filenames = orig_files + lora_images
    
    elif scenario == '+Diffusion':
        orig_features, orig_labels, orig_files = load_features(feature_paths[method_name])
        
        diff_dir = Path('/content/MO433/Trabalho Final/main/generated_images_diffusion_corel')
        diff_images = []
        diff_labels = []
        
        for class_id in range(1, 7):
            class_name = list(diff_dir.glob(f'*'))[class_id-1].name
            class_dir = diff_dir / class_name
            class_images = sorted(list(class_dir.glob('*.png')))
            
            for img_file in class_images:
                if 'diffusion' in img_file.name:
                    diff_images.append(img_file.name)
                    diff_labels.append(class_id)
        
        features = np.concatenate([orig_features, orig_features[:len(diff_images)]])
        labels = np.concatenate([orig_labels, np.array(diff_labels)])
        filenames = orig_files + diff_images
    
    
    return features, labels, filenames


def compute_clustering_metrics(features, true_labels, n_clusters=6):
    """Compute clustering metrics"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(features_scaled)
    
    silhouette = silhouette_score(features_scaled, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    return {
        'silhouette': silhouette,
        'ari': ari,
        'nmi': nmi,
        'pred_labels': pred_labels,
        'features_scaled': features_scaled
    }


def compute_umap_embedding(features, random_state=42):
    """Compute UMAP embedding"""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state,
        verbose=False
    )
    
    embedding = reducer.fit_transform(features)
    return embedding


def compute_tsne_embedding(features, random_state=42):
    """Compute t-SNE embedding"""
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=30,
        n_iter=1000
    )
    
    embedding = tsne.fit_transform(features)
    return embedding


def plot_embedding(embedding, labels, title, output_path, method='UMAP'):
    """Plot embedding"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for class_id in sorted(np.unique(labels)):
        mask = labels == class_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=CLASS_COLORS[class_id],
            label=CLASS_NAMES[class_id],
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method} Component 1', fontsize=12)
    ax.set_ylabel(f'{method} Component 2', fontsize=12)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_path}")


def create_comparison_plot(all_results, output_path):
    """Create comprehensive comparison plot"""
    methods = ['BYOL', 'CNN-JEPA', 'DGAE']
    scenarios = ['Original', '+LoRA', '+Diffusion']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    for i, method in enumerate(methods):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]
            
            key = f"{method}_{scenario}"
            if key not in all_results:
                ax.axis('off')
                continue
            
            result = all_results[key]
            embedding = result['umap_embedding']
            labels = result['true_labels']
            
            for class_id in sorted(np.unique(labels)):
                mask = labels == class_id
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=CLASS_COLORS[class_id],
                    label=CLASS_NAMES[class_id] if i == 0 and j == 0 else "",
                    alpha=0.6,
                    s=10,
                    edgecolors='none'
                )
            
            metrics = result['metrics']
            title = f"{method} - {scenario}\n"
            title += f"Sil: {metrics['silhouette']:.3f} | "
            title += f"ARI: {metrics['ari']:.3f} | "
            title += f"NMI: {metrics['nmi']:.3f}"
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.2)
    
    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_legend, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98), ncol=6, fontsize=10, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison plot: {output_path}")


def save_results_csv(all_results, output_path):
    """Save results to CSV"""
    rows = []
    
    for key, result in all_results.items():
        method, scenario = key.split('_', 1)
        metrics = result['metrics']
        
        rows.append({
            'Method': method,
            'Scenario': scenario,
            'Silhouette': metrics['silhouette'],
            'ARI': metrics['ari'],
            'NMI': metrics['nmi'],
            'Num_Samples': len(result['true_labels'])
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['Method', 'Scenario'])
    df.to_csv(output_path, index=False)
    
    print(f"Saved CSV results: {output_path}")
    
    return df


def save_results_json(all_results, output_path):
    """Save detailed results to JSON"""
    json_data = {}
    
    for key, result in all_results.items():
        json_data[key] = {
            'metrics': {
                'silhouette': float(result['metrics']['silhouette']),
                'ari': float(result['metrics']['ari']),
                'nmi': float(result['metrics']['nmi'])
            },
            'num_samples': int(len(result['true_labels'])),
            'num_classes': int(len(np.unique(result['true_labels'])))
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved JSON results: {output_path}")


def print_results_summary(df):
    """Print results summary"""

    print("BEST RESULTS BY METRIC")
    
    for metric in ['Silhouette', 'ARI', 'NMI']:
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        print(f"\n{metric}:")
        print(f"  {best_row['Method']} - {best_row['Scenario']}: {best_row[metric]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Clustering')
    parser.add_argument('--byol-features', type=str, default='/content/MO433/Trabalho Final/main/byol_features.pkl')
    parser.add_argument('--jepa-features', type=str, default='/content/MO433/Trabalho Final/main/jepa_features.pkl')
    parser.add_argument('--dgae-features', type=str, default='/content/MO433/Trabalho Final/main/dgae_features.pkl')
    parser.add_argument('--output-dir', type=str, default='/content/MO433/Trabalho Final/main/clustering_results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_paths = {
        'BYOL': args.byol_features,
        'CNN-JEPA': args.jepa_features,
        'DGAE': args.dgae_features
    }
    
    methods = ['BYOL', 'CNN-JEPA', 'DGAE']
    scenarios = ['Original', '+LoRA', '+Diffusion']
    
    all_results = {}
    
    for method in methods:
        if not Path(feature_paths[method]).exists():
            
            continue
        
        for scenario in scenarios:
            
            
            try:
                features, labels, filenames = load_scenario_features(method, scenario, feature_paths)
                
                metrics_result = compute_clustering_metrics(features, labels)
                
                umap_embedding = compute_umap_embedding(metrics_result['features_scaled'])
                
                tsne_embedding = compute_tsne_embedding(metrics_result['features_scaled'])
                
                key = f"{method}_{scenario}"
                all_results[key] = {
                    'metrics': {k: v for k, v in metrics_result.items() 
                               if k not in ['pred_labels', 'features_scaled']},
                    'umap_embedding': umap_embedding,
                    'tsne_embedding': tsne_embedding,
                    'true_labels': labels,
                    'num_samples': len(labels)
                }
                
                print(f"\nMetrics:")
                print(f"  Silhouette Score: {metrics_result['silhouette']:.4f}")
                print(f"  Adjusted Rand Index: {metrics_result['ari']:.4f}")
                print(f"  Normalized Mutual Info: {metrics_result['nmi']:.4f}")
                
                plot_embedding(
                    umap_embedding, labels,
                    f"{method} - {scenario} (UMAP)",
                    output_dir / f"{method.lower().replace('-', '_')}_{scenario.lower().replace('+', 'with_')}_umap.png",
                    method='UMAP'
                )
                
                plot_embedding(
                    tsne_embedding, labels,
                    f"{method} - {scenario} (t-SNE)",
                    output_dir / f"{method.lower().replace('-', '_')}_{scenario.lower().replace('+', 'with_')}_tsne.png",
                    method='t-SNE'
                )
                
            except Exception as e:
                print(f"Error processing {method} - {scenario}: {e}")
                continue
    
    create_comparison_plot(all_results, output_dir / 'comparison_all_methods.png')
    
    df = save_results_csv(all_results, output_dir / 'clustering_metrics.csv')
    save_results_json(all_results, output_dir / 'clustering_metrics.json')
    
    print_results_summary(df)
    
    print("Clustering completed")


if __name__ == "__main__":
    main()
