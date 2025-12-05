#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract Latent Features from DGAE - Task 4
Prepares features for clustering evaluation in Task 5
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle

class Config:
    data_dir = "/content/MO433/Trabalho Final/main/data/corel"
    output_dir = "/content/MO433/Trabalho Final/main/dgae_model"
    lora_dir = "/content/MO433/Trabalho Final/main/corel_lora_model"
    
    image_size = 256
    image_channels = 3
    latent_dim = 128
    hidden_dims = [64, 128, 256, 512]
    
    num_epochs = 200
    batch_size = 16
    learning_rate = 1e-4
    
    guidance_weight = 0.1
    recon_weight = 1.0
    
    weight_decay = 1e-5
    grad_clip = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    save_every = 20
    sample_every = 10
    seed = 42


class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.data_dir.glob(ext)))
            self.image_paths.extend(list(self.data_dir.glob(ext.upper())))
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        layers = []
        in_channels = config.image_channels
        
        for h_dim in config.hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
                ResidualBlock(h_dim),
            ))
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.final_channels = config.hidden_dims[-1]
        flatten_dim = self.final_channels * self.final_size * self.final_size
        
        self.fc = nn.Linear(flatten_dim, config.latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.final_channels = config.hidden_dims[-1]
        
        self.decoder_input = nn.Linear(config.latent_dim, 
                                       self.final_channels * self.final_size * self.final_size)
        
        layers = []
        reversed_dims = list(reversed(config.hidden_dims))
        
        for i in range(len(reversed_dims) - 1):
            layers.append(nn.Sequential(
                ResidualBlock(reversed_dims[i]),
                nn.Conv2d(reversed_dims[i], reversed_dims[i+1] * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(reversed_dims[i+1]),
                nn.LeakyReLU(0.2),
            ))
        
        layers.append(nn.Sequential(
            ResidualBlock(reversed_dims[-1]),
            nn.Conv2d(reversed_dims[-1], config.image_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Tanh()
        ))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.final_channels, self.final_size, self.final_size)
        return self.decoder(h)


class DGAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def load_dgae_model(checkpoint_path, device):
    print(f"Loading DGAE from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = DGAE(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ DGAE loaded successfully!")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Training epoch: {checkpoint['epoch']}")
    print(f"  Training loss: {checkpoint['loss']:.4f}")
    
    return model, config


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    
    all_features = []
    all_labels = []
    all_filenames = []
    
    print("Extracting latent features...")
    
    for images, filenames in tqdm(dataloader, desc="Processing"):
        images = images.to(device)
        features = model.encode(images)
        
        all_features.append(features.cpu().numpy())
        
        for filename in filenames:
            class_id = filename.split('_')[0]
            all_labels.append(int(class_id))
            all_filenames.append(filename)
    
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    print(f"✓ Extracted features from {len(all_features)} images")
    print(f"  Feature shape: {all_features.shape}")
    print(f"  Unique classes: {np.unique(all_labels)}")
    
    return all_features, all_labels, all_filenames


def save_features(features, labels, filenames, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'features': features,
        'labels': labels,
        'filenames': filenames,
        'feature_dim': features.shape[1],
        'num_samples': len(features),
        'num_classes': len(np.unique(labels))
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Features saved to: {output_path}")
    print(f"  Feature dimension: {data['feature_dim']}")
    print(f"  Number of samples: {data['num_samples']}")
    print(f"  Number of classes: {data['num_classes']}")


def print_class_distribution(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label:04d}: {count} images")


def main():
    parser = argparse.ArgumentParser(description='Extract features from DGAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to DGAE checkpoint')
    parser.add_argument('--data-dir', type=str, default='/content/MO433/Trabalho Final/main/data/corel',
                       help='Directory containing images')
    parser.add_argument('--output', type=str, default='/content/MO433/Trabalho Final/main/dgae_features.pkl',
                       help='Output path for extracted features')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    print("="*80)
    print("DGAE FEATURE EXTRACTION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Output:     {args.output}")
    print(f"Device:     {device}")
    print("="*80 + "\n")
    
    model, config = load_dgae_model(args.checkpoint, device)
    
    dataset = ImageDataset(args.data_dir, config.image_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"✓ Dataset: {len(dataset)} images\n")
    
    features, labels, filenames = extract_features(model, dataloader, device)
    
    print_class_distribution(labels)
    
    save_features(features, labels, filenames, args.output)
    
    print("\n" + "="*80)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("="*80)
    print(f"Features ready for clustering evaluation in Task 5")
    print("="*80)


if __name__ == "__main__":
    main()
