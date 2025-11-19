#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean VAE - No k-NN Complexity
================================

Focuses on what actually works:
- Proper KL control
- Perceptual loss for sharpness
- Good architecture
- Simple and fast
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse


class Config:
    data_dir = "training_data/dogs"
    output_dir = "./vae_clean"
    image_size = 128
    image_channels = 3
    latent_dim = 128  # Larger for better capacity
    hidden_dims = [64, 128, 256, 512]
    num_epochs = 300
    batch_size = 16
    learning_rate = 1e-4
    
    # KL control (OPTIMIZED from experiments)
    kl_weight_final = 1.0     # Proper balance (was 0.01 - too small!)
    kl_warmup_epochs = 150    # Gentle warmup for stability
    kl_target = 25.0
    
    # Perceptual loss (balanced)
    use_perceptual = True
    perceptual_weight = 0.03  # Reduced from 0.05 for better KL/recon balance
    
    weight_decay = 1e-5
    grad_clip = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    save_every = 20
    sample_every = 10
    seed = 42


config = Config()


class SimpleDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
            self.image_paths.extend(list(self.data_dir.rglob(ext.upper())))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"✓ Found {len(self.image_paths)} images")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss - THE KEY TO SHARPNESS"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = vgg.features[:16].eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        x = (x + 1) / 2
        return (x - self.mean) / self.std
    
    def forward(self, x, y):
        x_norm = self.normalize(x)
        y_norm = self.normalize(y)
        x_features = self.feature_extractor(x_norm)
        y_features = self.feature_extractor(y_norm)
        return F.mse_loss(x_features, y_features)


class ResidualBlock(nn.Module):
    """Simple residual block for better features"""
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
        
        self.fc_mu = nn.Linear(flatten_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, config.latent_dim)
        
        # CRITICAL FIX: Initialize logvar to start with low variance
        # This prevents KL explosion (4500+) at epoch 1
        # Strategy: Small weights + negative bias → logvar starts around -3
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.zeros_(self.fc_mu.bias)
        
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)
        nn.init.constant_(self.fc_logvar.bias, -3.0)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


class Decoder(nn.Module):
    """PixelShuffle decoder with residual blocks"""
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


class CleanVAE(nn.Module):
    """Simple, effective VAE without k-NN complexity"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        if config.use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, model, beta=1.0):
    """Simple, effective loss with PROPER KL control"""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # Perceptual loss (key to sharpness!)
    if model.config.use_perceptual:
        perceptual = model.perceptual_loss(recon_x, x)
        recon_loss = recon_loss + model.config.perceptual_weight * perceptual
    else:
        perceptual = torch.tensor(0.0)
    
    # KL divergence - STANDARD FORMULATION
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = torch.mean(kl_div)
    
    # Simple direct penalty (not adaptive)
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div, perceptual


def get_kl_weight(epoch, current_kl, config):
    """Simple linear warmup - no fancy adaptation"""
    if epoch < config.kl_warmup_epochs:
        return config.kl_weight_final * (epoch / config.kl_warmup_epochs)
    else:
        return config.kl_weight_final


def train_epoch(model, dataloader, optimizer, config, epoch, kl_history):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_perceptual = 0
    num_batches = 0
    
    recent_kl = np.mean(kl_history[-10:]) if len(kl_history) > 0 else 50.0
    current_beta = get_kl_weight(epoch, recent_kl, config)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for data in pbar:
        data = data.to(config.device)
        optimizer.zero_grad()
        
        recon, mu, logvar = model(data)
        loss, recon_loss, kl_div, perceptual = vae_loss(
            recon, data, mu, logvar, model, beta=current_beta
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_div.item()
        total_perceptual += perceptual.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_div.item():.1f}',
            'perc': f'{perceptual.item():.4f}'
        })
    
    avg_kl = total_kl / num_batches
    return (total_loss / num_batches, total_recon / num_batches, 
            avg_kl, current_beta, total_perceptual / num_batches)


@torch.no_grad()
def generate_samples(model, epoch, output_dir, device, num_samples=16):
    model.eval()
    z = torch.randn(num_samples, model.latent_dim).to(device)
    samples = model.decode(z)
    
    samples_dir = Path(output_dir) / 'samples'
    samples_dir.mkdir(exist_ok=True, parents=True)
    save_image(samples, samples_dir / f'samples_epoch_{epoch}.png', 
               nrow=4, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def visualize_reconstruction(model, dataloader, epoch, output_dir, device, num_images=8):
    model.eval()
    data = next(iter(dataloader))[:num_images].to(device)
    recon, mu, logvar = model(data)
    
    comparison = torch.cat([data, recon])
    
    recon_dir = Path(output_dir) / 'reconstructions'
    recon_dir.mkdir(exist_ok=True, parents=True)
    save_image(comparison, recon_dir / f'reconstruction_epoch_{epoch}.png',
               nrow=num_images, normalize=True, value_range=(-1, 1))


def plot_losses(losses, output_dir, config):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(losses['total'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(losses['recon'])
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(losses['kl'])
    axes[1, 0].axhline(y=config.kl_target, color='g', linestyle='--', label='Target')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(losses['perceptual'])
    axes[1, 1].set_title('Perceptual Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_losses.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Clean VAE - No k-NN Complexity')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--kl-weight', type=float, default=0.01)
    parser.add_argument('--kl-warmup', type=int, default=50)
    parser.add_argument('--kl-target', type=float, default=25.0)
    parser.add_argument('--no-perceptual', action='store_true')
    parser.add_argument('--perceptual-weight', type=float, default=0.05)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./vae_clean')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    config.data_dir = args.data_dir
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.latent_dim = args.latent_dim
    config.learning_rate = args.learning_rate
    config.kl_weight_final = args.kl_weight
    config.kl_warmup_epochs = args.kl_warmup
    config.kl_target = args.kl_target
    config.use_perceptual = not args.no_perceptual
    config.perceptual_weight = args.perceptual_weight
    config.output_dir = args.output_dir
    config.seed = args.seed
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print("="*80)
    print("CLEAN VAE - SIMPLE & EFFECTIVE")
    print("="*80)
    print(f"Data:            {config.data_dir}")
    print(f"Latent dim:      {config.latent_dim}")
    print(f"KL target:       {config.kl_target}")
    print(f"Perceptual:      {'YES' if config.use_perceptual else 'NO'}")
    print(f"Epochs:          {config.num_epochs}")
    print("="*80 + "\n")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    dataset = SimpleDataset(config.data_dir, config.image_size)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, 
        pin_memory=True if config.device == "cuda" else False,
        drop_last=True
    )
    
    print(f"✓ Dataset: {len(dataset)} images\n")
    
    model = CleanVAE(config).to(config.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Parameters: {num_params:,}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                 weight_decay=config.weight_decay)
    
    losses = {'total': [], 'recon': [], 'kl': [], 'beta': [], 'perceptual': []}
    kl_history = []
    best_loss = float('inf')
    start_epoch = 0
    
    if args.resume:
        print(f"Loading: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        best_loss = checkpoint['loss']
        print(f"✓ Resumed from epoch {start_epoch}\n")
    
    print("="*80)
    print("Training...")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, config.num_epochs):
        avg_loss, avg_recon, avg_kl, beta, avg_perc = train_epoch(
            model, dataloader, optimizer, config, epoch, kl_history
        )
        
        kl_history.append(avg_kl)
        losses['total'].append(avg_loss)
        losses['recon'].append(avg_recon)
        losses['kl'].append(avg_kl)
        losses['beta'].append(beta)
        losses['perceptual'].append(avg_perc)
        
        kl_status = "✓" if abs(avg_kl - config.kl_target) <= 10 else "⚠"
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Loss:  {avg_loss:.4f}")
        print(f"  Recon: {avg_recon:.4f}")
        print(f"  Perc:  {avg_perc:.4f}")
        print(f"  KL:    {avg_kl:.1f} {kl_status}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
            }, Path(config.output_dir) / 'best_model.pt')
            print(f"  ✓ Saved")
        
        if (epoch + 1) % config.sample_every == 0:
            generate_samples(model, epoch + 1, config.output_dir, config.device, 16)
            visualize_reconstruction(model, dataloader, epoch + 1, config.output_dir, config.device)
            plot_losses(losses, config.output_dir, config)
        
        if (epoch + 1) % config.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
            }, Path(config.output_dir) / f'checkpoint_{epoch+1:04d}.pt')
    
    print("\n" + "="*80)
    print("DONE!")
    print(f"Final KL: {losses['kl'][-1]:.1f}")
    print("="*80)


if __name__ == "__main__":
    main()
