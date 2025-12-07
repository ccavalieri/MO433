#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering Evaluation - Task 5 - VERSÃO CORRIGIDA
FIX: Agora extrai features REAIS das imagens sintéticas ao invés de duplicar features originais
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
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


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


class SyntheticImageDataset(Dataset):
    """Dataset para carregar imagens sintéticas"""
    def __init__(self, image_paths, image_size=256):
        self.image_paths = image_paths
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), str(img_path.name)


def extract_features_from_images(model, image_paths, method_name, device='cuda', batch_size=32):
    """
    Extrai features de imagens usando o modelo treinado
    
    FIX CRÍTICO: Esta função agora realmente carrega e processa as imagens sintéticas!
    """
    dataset = SyntheticImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model.eval()
    all_features = []
    all_filenames = []
    
    print(f"  Extraindo features de {len(image_paths)} imagens sintéticas...")
    
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Processing"):
            images = images.to(device)
            
            # Extrai features dependendo do método
            if method_name == 'BYOL':
                features = model.encode_only(images)
                features = features.view(features.size(0), -1).cpu().numpy()
            elif method_name == 'CNN-JEPA':
                features = model.encode_only(images).cpu().numpy()
            elif method_name == 'DGAE':
                features = model.encode(images).cpu().numpy()
            else:
                raise ValueError(f"Método desconhecido: {method_name}")
            
            all_features.append(features)
            all_filenames.extend(filenames)
    
    all_features = np.vstack(all_features)
    print(f"  ✓ Features extraídas: {all_features.shape}")
    
    return all_features, all_filenames


def load_model_for_inference(method_name, checkpoint_path, device='cuda'):
    """Carrega modelo treinado para extração de features"""
    print(f"  Carregando modelo {method_name} de {checkpoint_path}")
    
    if method_name == 'BYOL':
        from main.train_byol_corel import BYOLModel
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = BYOLModel(input_channels=3, image_size=256, projection_dim=128).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    elif method_name == 'CNN-JEPA':
        from main.train_cnn_jepa_corel import CNNJEPAModel
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = CNNJEPAModel(input_channels=3, embed_dim=256).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    elif method_name == 'DGAE':
        from main.extract_dgae_features import DGAE
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        model = DGAE(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    else:
        raise ValueError(f"Método desconhecido: {method_name}")
    
    model.eval()
    print(f"  ✓ Modelo carregado")
    return model


def load_scenario_features_FIXED(method_name, scenario, feature_paths, model_checkpoints, device='cuda'):
    """
    VERSÃO CORRIGIDA: Carrega features para um método e cenário específico
    
    FIX: Agora realmente extrai features das imagens sintéticas!
    """
    print(f"\n{'='*60}")
    print(f"Loading features: {method_name} - {scenario}")
    print(f"{'='*60}")
    
    # Carrega features originais
    orig_features, orig_labels, orig_files = load_features(feature_paths[method_name])
    print(f"  Original dataset: {len(orig_features)} imagens")
    
    if scenario == 'Original':
        return orig_features, orig_labels, orig_files
    
    # Para cenários com sintéticas, carrega o modelo e extrai features REAIS
    model = load_model_for_inference(method_name, model_checkpoints[method_name], device)
    
    if scenario == '+LoRA':
        lora_dir = Path('./generated_images_corel')
        synthetic_paths = []
        synthetic_labels = []
        
        for class_id in range(1, 7):
            # Busca diretório da classe
            class_dirs = sorted([d for d in lora_dir.iterdir() if d.is_dir()])
            if class_id - 1 < len(class_dirs):
                class_dir = class_dirs[class_id - 1]
                class_images = sorted(list(class_dir.glob('*synthetic.png')))
                
                for img_path in class_images:
                    synthetic_paths.append(img_path)
                    synthetic_labels.append(class_id)
        
        print(f"  Encontradas {len(synthetic_paths)} imagens sintéticas LoRA")
        
        # EXTRAI FEATURES REAIS DAS IMAGENS SINTÉTICAS!
        synthetic_features, synthetic_files = extract_features_from_images(
            model, synthetic_paths, method_name, device
        )
        
    elif scenario == '+Diffusion':
        diff_dir = Path('./generated_images_diffusion_corel')
        synthetic_paths = []
        synthetic_labels = []
        
        for class_id in range(1, 7):
            class_dirs = sorted([d for d in diff_dir.iterdir() if d.is_dir()])
            if class_id - 1 < len(class_dirs):
                class_dir = class_dirs[class_id - 1]
                class_images = sorted(list(class_dir.glob('*diffusion.png')))
                
                for img_path in class_images:
                    synthetic_paths.append(img_path)
                    synthetic_labels.append(class_id)
        
        print(f"  Encontradas {len(synthetic_paths)} imagens sintéticas Diffusion")
        
        # EXTRAI FEATURES REAIS DAS IMAGENS SINTÉTICAS!
        synthetic_features, synthetic_files = extract_features_from_images(
            model, synthetic_paths, method_name, device
        )
    
    # Combina features originais + sintéticas
    features = np.concatenate([orig_features, synthetic_features])
    labels = np.concatenate([orig_labels, np.array(synthetic_labels)])
    filenames = orig_files + synthetic_files
    
    print(f"  Dataset combinado: {len(features)} imagens")
    print(f"    - Originais: {len(orig_features)}")
    print(f"    - Sintéticas: {len(synthetic_features)}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
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


def plot_embedding(embedding, labels, title, output_path, method='UMAP'):
    """Plot embedding with fixed colors"""
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
    
    print(f"\n✓ Saved comparison plot: {output_path}")


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
    
    print(f"✓ Saved CSV results: {output_path}")
    
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
    
    print(f"✓ Saved JSON results: {output_path}")


def print_results_summary(df):
    """Print results summary"""
    print("\n" + "="*80)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    print("\n" + "="*80)
    print("BEST RESULTS BY METRIC")
    print("="*80)
    
    for metric in ['Silhouette', 'ARI', 'NMI']:
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        print(f"\n{metric}:")
        print(f"  {best_row['Method']} - {best_row['Scenario']}: {best_row[metric]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Clustering - VERSÃO CORRIGIDA')
    parser.add_argument('--byol-features', type=str, default='/content/MO433/Trabalho Final/main/byol_features.pkl')
    parser.add_argument('--jepa-features', type=str, default='/content/MO433/Trabalho Final/main/jepa_features.pkl')
    parser.add_argument('--dgae-features', type=str, default='/content/MO433/Trabalho Final/main/dgae_features.pkl')
    parser.add_argument('--byol-checkpoint', type=str, default='/content/MO433/Trabalho Final/main/byol_model/best_model.pt')
    parser.add_argument('--jepa-checkpoint', type=str, default='/content/MO433/Trabalho Final/main/jepa_model/best_model.pt')
    parser.add_argument('--dgae-checkpoint', type=str, default='/content/MO433/Trabalho Final/main/dgae_model/best_model.pt')
    parser.add_argument('--output-dir', type=str, default='/content/MO433/Trabalho Final/main/clustering_results')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_paths = {
        'BYOL': args.byol_features,
        'CNN-JEPA': args.jepa_features,
        'DGAE': args.dgae_features
    }
    
    model_checkpoints = {
        'BYOL': args.byol_checkpoint,
        'CNN-JEPA': args.jepa_checkpoint,
        'DGAE': args.dgae_checkpoint
    }
    
    print("="*80)
    print("CLUSTERING EVALUATION - VERSÃO CORRIGIDA")
    print("="*80)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    print("\n🔧 FIX: Agora extrai features REAIS das imagens sintéticas!")
    print("="*80)
    
    methods = ['BYOL', 'CNN-JEPA', 'DGAE']
    scenarios = ['Original', '+LoRA', '+Diffusion']
    
    all_results = {}
    
    for method in methods:
        if not Path(feature_paths[method]).exists():
            print(f"\n⚠ Warning: Features not found for {method}, skipping...")
            continue
        
        if not Path(model_checkpoints[method]).exists():
            print(f"\n⚠ Warning: Checkpoint not found for {method}, skipping synthetic scenarios...")
            scenarios_to_eval = ['Original']
        else:
            scenarios_to_eval = scenarios
        
        for scenario in scenarios_to_eval:
            try:
                features, labels, filenames = load_scenario_features_FIXED(
                    method, scenario, feature_paths, model_checkpoints, device
                )
                
                print("\n Computing clustering metrics...")
                metrics_result = compute_clustering_metrics(features, labels)
                
                print("Computing UMAP embedding...")
                umap_embedding = compute_umap_embedding(metrics_result['features_scaled'])
                
                key = f"{method}_{scenario}"
                all_results[key] = {
                    'metrics': {k: v for k, v in metrics_result.items() 
                               if k not in ['pred_labels', 'features_scaled']},
                    'umap_embedding': umap_embedding,
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
                
            except Exception as e:
                print(f"✗ Error processing {method} - {scenario}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(all_results) == 0:
        print("\n✗ No results to save!")
        return
    
    print(f"\n{'='*80}")
    print("Creating comparison visualizations...")
    print(f"{'='*80}")
    
    create_comparison_plot(all_results, output_dir / 'comparison_all_methods_FIXED.png')
    
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    df = save_results_csv(all_results, output_dir / 'clustering_metrics_FIXED.csv')
    save_results_json(all_results, output_dir / 'clustering_metrics_FIXED.json')
    
    print_results_summary(df)
    
    print("\n" + "="*80)
    print("✅ CLUSTERING EVALUATION COMPLETE!")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()