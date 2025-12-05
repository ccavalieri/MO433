#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train BYOL on Corel Dataset and Extract Features
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
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.augment:
            return self.base_transform(image)
        else:
            return self.base_transform(image), str(img_path.name)


def get_byol_augmentation(image_size=256):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize([0.5], [0.5])
    ])


def create_byol_pairs(batch, augment_fn):
    batch_size = batch.shape[0]
    
    augmented_1 = []
    augmented_2 = []
    
    for img in batch:
        img_denorm = (img * 0.5 + 0.5)
        
        aug1 = augment_fn(img_denorm)
        aug2 = augment_fn(img_denorm)
        
        augmented_1.append(aug1)
        augmented_2.append(aug2)
    
    augmented_1 = torch.stack(augmented_1)
    augmented_2 = torch.stack(augmented_2)
    
    return augmented_1, augmented_2


def create_encoder(input_channels=3):
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, 3, stride=2, padding=1), 
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.Conv2d(128, 256, 3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
    )


def create_projector(feature_dim, projection_dim=128):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(True),
        nn.Linear(512, projection_dim)
    )


def create_predictor(projection_dim=128):
    return nn.Sequential(
        nn.Linear(projection_dim, projection_dim//2),
        nn.BatchNorm1d(projection_dim//2),
        nn.ReLU(True),
        nn.Linear(projection_dim//2, projection_dim)
    )


class BYOLModel(nn.Module):
    def __init__(self, input_channels=3, image_size=256, projection_dim=128):
        super().__init__()
        
        encoder_out_size = image_size // 16
        self.feature_dim = 256 * encoder_out_size * encoder_out_size
        
        self.online_encoder = create_encoder(input_channels)
        self.online_projector = create_projector(self.feature_dim, projection_dim)
        self.predictor = create_predictor(projection_dim)
        
        self.target_encoder = create_encoder(input_channels)
        self.target_projector = create_projector(self.feature_dim, projection_dim)
        
        self._initialize_target_network()
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def _initialize_target_network(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), 
                                            self.target_encoder.parameters()):
            target_param.data.copy_(online_param.data)
        
        for online_param, target_param in zip(self.online_projector.parameters(), 
                                            self.target_projector.parameters()):
            target_param.data.copy_(online_param.data)
    
    def update_target_network(self, momentum=0.996):
        for online_param, target_param in zip(self.online_encoder.parameters(), 
                                            self.target_encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * online_param.data
        
        for online_param, target_param in zip(self.online_projector.parameters(), 
                                            self.target_projector.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * online_param.data
    
    def forward(self, x1, x2):
        online_features1 = self.online_encoder(x1)
        online_features2 = self.online_encoder(x2)
        
        online_proj1 = self.online_projector(online_features1)
        online_proj2 = self.online_projector(online_features2)
        
        online_pred1 = self.predictor(online_proj1)
        online_pred2 = self.predictor(online_proj2)
        
        with torch.no_grad():
            target_features1 = self.target_encoder(x1)
            target_features2 = self.target_encoder(x2)
            
            target_proj1 = self.target_projector(target_features1)
            target_proj2 = self.target_projector(target_features2)
        
        return online_pred1, online_pred2, target_proj1, target_proj2, online_features1
    
    def encode_only(self, x):
        features = self.online_encoder(x)
        return features


def byol_loss(online_pred1, online_pred2, target_proj1, target_proj2):
    online_pred1 = F.normalize(online_pred1, dim=1)
    online_pred2 = F.normalize(online_pred2, dim=1) 
    target_proj1 = F.normalize(target_proj1, dim=1)
    target_proj2 = F.normalize(target_proj2, dim=1)
    
    loss1 = 2 - 2 * (online_pred1 * target_proj2).sum(dim=1).mean()
    loss2 = 2 - 2 * (online_pred2 * target_proj1).sum(dim=1).mean()
    
    total_loss = (loss1 + loss2) / 2
    
    return total_loss


def train_batch_byol(data, model, optimizer, augment_fn, momentum, device):
    model.train()
    data = data.to(device)
    
    aug1, aug2 = create_byol_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    optimizer.zero_grad()
    
    online_pred1, online_pred2, target_proj1, target_proj2, _ = model(aug1, aug2)
    
    loss = byol_loss(online_pred1, online_pred2, target_proj1, target_proj2)
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    model.update_target_network(momentum)
    
    return loss


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    
    all_features = []
    all_labels = []
    all_filenames = []
    
    print("Extracting features")
    
    for data, filenames in tqdm(dataloader, desc="Processing"):
        data = data.to(device)
        
        features = model.encode_only(data)
        features_flat = features.view(features.size(0), -1).cpu().numpy()
        
        all_features.append(features_flat)
        
        for filename in filenames:
            class_id = filename.split('_')[0]
            all_labels.append(int(class_id))
            all_filenames.append(filename)
    
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    print(f"Extracted features from {len(all_features)} images")
    
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
    
    print(f"Features saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train BYOL on Corel Dataset')
    parser.add_argument('--data-dir', type=str, default='/content/MO433/Trabalho Final/main/data/corel')
    parser.add_argument('--output-dir', type=str, default='/content/MO433/Trabalho Final/main/byol_model')
    parser.add_argument('--features-output', type=str, default='/content/MO433/Trabalho Final/main/byol_features.pkl')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("BYOL TRAINING - COREL DATASET")
    print(f"Data:        {args.data_dir}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Image size:  {args.image_size}")
    print(f"Device:      {device}")
    
    train_dataset = CorelDataset(args.data_dir, args.image_size, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"Dataset: {len(train_dataset)} images\n")
    
    model = BYOLModel(input_channels=3, image_size=args.image_size, projection_dim=128).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    augment_fn = get_byol_augmentation(args.image_size)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        momentum = 0.99 + (0.999 - 0.99) * (epoch / args.epochs)
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for data in pbar:
            loss = train_batch_byol(data, model, optimizer, augment_fn, momentum, device)
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
            print(f"  Saved")
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, Path(args.output_dir) / f'checkpoint_{epoch+1:04d}.pt')
    
    print("Extracting features")
    
    eval_dataset = CorelDataset(args.data_dir, args.image_size, augment=False)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    features, labels, filenames = extract_features(model, eval_loader, device)
    save_features(features, labels, filenames, args.features_output)
    
    print("Done")


if __name__ == "__main__":
    main()
