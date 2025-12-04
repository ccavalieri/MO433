#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train CNN-JEPA on Corel Dataset and Extract Features - Task 5
Joint Embedding Predictive Architecture for CNNs
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pickle
import argparse


class CorelDataset(Dataset):
    def __init__(self, data_dir, image_size=256, augment=False):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.data_dir.glob(ext)))
            self.image_paths.extend(list(self.data_dir.glob(ext.upper())))
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.augment:
            image = self.augment_transform(image)
        
        tensor = self.base_transform(image)
        
        if self.augment:
            return tensor
        else:
            return tensor, str(img_path.name)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ContextEncoder(nn.Module):
    """Context encoder for CNN-JEPA"""
    def __init__(self, input_channels=3, embed_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, embed_dim, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)
        
        if return_features:
            return features
        
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        
        return x, features


class Predictor(nn.Module):
    """Predictor network for CNN-JEPA"""
    def __init__(self, embed_dim=256, hidden_dim=512):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


class CNNJEPAModel(nn.Module):
    """CNN-JEPA: Joint Embedding Predictive Architecture"""
    def __init__(self, input_channels=3, embed_dim=256):
        super().__init__()
        
        self.context_encoder = ContextEncoder(input_channels, embed_dim)
        self.target_encoder = ContextEncoder(input_channels, embed_dim)
        self.predictor = Predictor(embed_dim)
        
        self._initialize_target_encoder()
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    def _initialize_target_encoder(self):
        for param_c, param_t in zip(self.context_encoder.parameters(), 
                                    self.target_encoder.parameters()):
            param_t.data.copy_(param_c.data)
    
    def update_target_encoder(self, momentum=0.996):
        for param_c, param_t in zip(self.context_encoder.parameters(), 
                                    self.target_encoder.parameters()):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_c.data
    
    def forward(self, x_context, x_target):
        context_embed, context_features = self.context_encoder(x_context)
        
        predicted_embed = self.predictor(context_embed)
        
        with torch.no_grad():
            target_embed, _ = self.target_encoder(x_target)
        
        return predicted_embed, target_embed, context_features
    
    def encode_only(self, x):
        """Extract features for evaluation"""
        with torch.no_grad():
            embed, features = self.context_encoder(x)
        return embed


def create_masked_views(images, mask_ratio=0.3):
    """Create context and target views with random masking"""
    batch_size, channels, height, width = images.shape
    
    context_images = images.clone()
    target_images = images.clone()
    
    mask_height = int(height * mask_ratio)
    mask_width = int(width * mask_ratio)
    
    for i in range(batch_size):
        y = np.random.randint(0, height - mask_height)
        x = np.random.randint(0, width - mask_width)
        
        context_images[i, :, y:y+mask_height, x:x+mask_width] = 0
    
    return context_images, target_images


def jepa_loss(predicted_embed, target_embed):
    """Variance-Invariance-Covariance loss for JEPA"""
    predicted_embed = F.normalize(predicted_embed, dim=1)
    target_embed = F.normalize(target_embed, dim=1)
    
    loss = 2 - 2 * (predicted_embed * target_embed).sum(dim=1).mean()
    
    return loss


def train_batch_jepa(data, model, optimizer, momentum, device):
    model.train()
    data = data.to(device)
    
    context_images, target_images = create_masked_views(data, mask_ratio=0.3)
    
    optimizer.zero_grad()
    
    predicted_embed, target_embed, _ = model(context_images, target_images)
    
    loss = jepa_loss(predicted_embed, target_embed)
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    model.update_target_encoder(momentum)
    
    return loss


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    
    all_features = []
    all_labels = []
    all_filenames = []
    
    print("Extracting features...")
    
    for data, filenames in tqdm(dataloader, desc="Processing"):
        data = data.to(device)
        
        features = model.encode_only(data).cpu().numpy()
        
        all_features.append(features)
        
        for filename in filenames:
            class_id = filename.split('_')[0]
            all_labels.append(int(class_id))
            all_filenames.append(filename)
    
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    print(f"✓ Extracted features from {len(all_features)} images")
    print(f"  Feature shape: {all_features.shape}")
    
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


def main():
    parser = argparse.ArgumentParser(description='Train CNN-JEPA on Corel Dataset')
    parser.add_argument('--data-dir', type=str, default='./data/corel')
    parser.add_argument('--output-dir', type=str, default='./jepa_model')
    parser.add_argument('--features-output', type=str, default='./jepa_features.pkl')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("CNN-JEPA TRAINING - COREL DATASET")
    print("="*80)
    print(f"Data:        {args.data_dir}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Image size:  {args.image_size}")
    print(f"Device:      {device}")
    print("="*80 + "\n")
    
    train_dataset = CorelDataset(args.data_dir, args.image_size, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"✓ Dataset: {len(train_dataset)} images\n")
    
    model = CNNJEPAModel(input_channels=3, embed_dim=256).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print("="*80)
    print("Training...")
    print("="*80 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        momentum = 0.99 + (0.999 - 0.99) * (epoch / args.epochs)
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for data in pbar:
            loss = train_batch_jepa(data, model, optimizer, momentum, device)
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  LR:   {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Momentum: {momentum:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, Path(args.output_dir) / 'best_model.pt')
            print(f"  ✓ Saved")
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, Path(args.output_dir) / f'checkpoint_{epoch+1:04d}.pt')
    
    print("\n" + "="*80)
    print("Training complete! Extracting features...")
    print("="*80 + "\n")
    
    eval_dataset = CorelDataset(args.data_dir, args.image_size, augment=False)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    features, labels, filenames = extract_features(model, eval_loader, device)
    save_features(features, labels, filenames, args.features_output)
    
    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
