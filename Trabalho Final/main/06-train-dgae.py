#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DGAE - Diffusion-Guided Autoencoder for Corel Dataset - Task 4
Uses Stable Diffusion + LoRA from Task 2 for guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DDPMScheduler
import argparse


class Config:
    data_dir = "./data/corel"
    output_dir = "./dgae_model"
    lora_dir = "./corel_lora_model"
    
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


config = Config()


class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.data_dir.glob(ext)))
            self.image_paths.extend(list(self.data_dir.glob(ext.upper())))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"✓ Found {len(self.image_paths)} images")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


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


def find_latest_lora(lora_dir):
    lora_path = Path(lora_dir)
    lora_files = list(lora_path.glob("*.safetensors"))
    
    if not lora_files:
        raise FileNotFoundError(f"No LoRA files found in {lora_dir}")
    
    latest_lora = max(lora_files, key=lambda x: x.stat().st_mtime)
    return latest_lora.name


def load_diffusion_model(lora_dir, device):
    print("Loading Stable Diffusion + LoRA for guidance...")
    
    lora_name = find_latest_lora(lora_dir)
    print(f"Found LoRA: {lora_name}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=lora_dir,
        weight_name=lora_name,
        adapter_name="corel_lora"
    )
    pipe.set_adapters(["corel_lora"], adapter_weights=[1.0])
    
    unet = pipe.unet
    vae = pipe.vae
    
    unet.eval()
    vae.eval()
    for param in unet.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    
    print("✓ Diffusion model loaded successfully!")
    
    return unet, vae


@torch.no_grad()
def extract_diffusion_features(images, unet, vae, device):
    """Extract intermediate features from diffusion model U-Net"""
    images = images.to(device, dtype=torch.float16)
    
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    
    timesteps = torch.randint(0, 1000, (images.shape[0],), device=device).long()
    noise = torch.randn_like(latents)
    noisy_latents = latents + noise * 0.1
    
    down_block_res_samples = []
    
    sample = noisy_latents
    emb = unet.time_embedding(timesteps)
    
    sample = unet.conv_in(sample)
    
    for downsample_block in unet.down_blocks:
        sample, res_samples = downsample_block(
            hidden_states=sample,
            temb=emb,
        )
        down_block_res_samples.extend(res_samples)
    
    features = torch.cat([f.mean(dim=[2, 3]) for f in down_block_res_samples[-3:]], dim=1)
    
    return features.float()


def dgae_loss(recon_x, x, guidance_features, guidance_weight=0.1, recon_weight=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    if guidance_features is not None:
        guidance_loss = guidance_features.mean()
    else:
        guidance_loss = torch.tensor(0.0, device=x.device)
    
    total_loss = recon_weight * recon_loss + guidance_weight * guidance_loss
    
    return total_loss, recon_loss, guidance_loss


def train_epoch(model, dataloader, optimizer, unet, vae, config, epoch):
    model.train()
    total_loss = 0
    total_recon = 0
    total_guidance = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for data in pbar:
        data = data.to(config.device)
        optimizer.zero_grad()
        
        with torch.no_grad():
            guidance_features = extract_diffusion_features(data, unet, vae, config.device)
        
        recon, z = model(data)
        
        loss, recon_loss, guidance_loss = dgae_loss(
            recon, data, guidance_features,
            guidance_weight=config.guidance_weight,
            recon_weight=config.recon_weight
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_guidance += guidance_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'guide': f'{guidance_loss.item():.4f}'
        })
    
    return (total_loss / num_batches, total_recon / num_batches, total_guidance / num_batches)


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
    recon, z = model(data)
    
    comparison = torch.cat([data, recon])
    
    recon_dir = Path(output_dir) / 'reconstructions'
    recon_dir.mkdir(exist_ok=True, parents=True)
    save_image(comparison, recon_dir / f'reconstruction_epoch_{epoch}.png',
               nrow=num_images, normalize=True, value_range=(-1, 1))


def plot_losses(losses, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(losses['total'])
    axes[0].set_title('Total Loss')
    axes[0].grid(True)
    
    axes[1].plot(losses['recon'])
    axes[1].set_title('Reconstruction Loss')
    axes[1].grid(True)
    
    axes[2].plot(losses['guidance'])
    axes[2].set_title('Guidance Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_losses.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train DGAE for Corel Dataset')
    parser.add_argument('--data-dir', type=str, default='./data/corel')
    parser.add_argument('--lora-dir', type=str, default='./corel_lora_model')
    parser.add_argument('--output-dir', type=str, default='./dgae_model')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--guidance-weight', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    config.data_dir = args.data_dir
    config.lora_dir = args.lora_dir
    config.output_dir = args.output_dir
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.guidance_weight = args.guidance_weight
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print("="*80)
    print("DGAE TRAINING - DIFFUSION-GUIDED AUTOENCODER")
    print("="*80)
    print(f"Data:            {config.data_dir}")
    print(f"LoRA:            {config.lora_dir}")
    print(f"Resolution:      {config.image_size}x{config.image_size}")
    print(f"Latent dim:      {config.latent_dim}")
    print(f"Epochs:          {config.num_epochs}")
    print(f"Batch size:      {config.batch_size}")
    print(f"Guidance weight: {config.guidance_weight}")
    print("="*80 + "\n")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    unet, vae = load_diffusion_model(config.lora_dir, config.device)
    
    dataset = ImageDataset(config.data_dir, config.image_size)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, 
        pin_memory=True if config.device == "cuda" else False,
        drop_last=True
    )
    
    print(f"✓ Dataset: {len(dataset)} images\n")
    
    model = DGAE(config).to(config.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                 weight_decay=config.weight_decay)
    
    losses = {'total': [], 'recon': [], 'guidance': []}
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
        avg_loss, avg_recon, avg_guidance = train_epoch(
            model, dataloader, optimizer, unet, vae, config, epoch
        )
        
        losses['total'].append(avg_loss)
        losses['recon'].append(avg_recon)
        losses['guidance'].append(avg_guidance)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Loss:     {avg_loss:.4f}")
        print(f"  Recon:    {avg_recon:.4f}")
        print(f"  Guidance: {avg_guidance:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
                'config': config,
            }, Path(config.output_dir) / 'best_model.pt')
            print(f"  ✓ Saved")
        
        if (epoch + 1) % config.sample_every == 0:
            generate_samples(model, epoch + 1, config.output_dir, config.device, 16)
            visualize_reconstruction(model, dataloader, epoch + 1, config.output_dir, config.device)
            plot_losses(losses, config.output_dir)
        
        if (epoch + 1) % config.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
                'config': config,
            }, Path(config.output_dir) / f'checkpoint_{epoch+1:04d}.pt')
    
    print("\n" + "="*80)
    print("DONE!")
    print(f"Best loss: {best_loss:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
