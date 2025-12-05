#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Latent Diffusion Model for Corel Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from collections import deque
from pathlib import Path


# VAE Components
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
        
        self.fc_mu = nn.Linear(flatten_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, config.latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


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


class VAEConfig:
    image_size = 256
    image_channels = 3
    latent_dim = 128
    hidden_dims = [64, 128, 256, 512]


def load_vae_components(vae_checkpoint_path, device):
    print(f"Loading VAE from: {vae_checkpoint_path}")
    
    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    vae_config = VAEConfig()
    
    encoder = Encoder(vae_config).to(device)
    decoder = Decoder(vae_config).to(device)
    
    state_dict = checkpoint['model_state_dict']
    
    encoder_state = {}
    decoder_state = {}
    
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            new_key = k[8:]
            encoder_state[new_key] = v
        elif k.startswith('decoder.'):
            new_key = k[8:]
            decoder_state[new_key] = v
    
    encoder.load_state_dict(encoder_state, strict=False)
    decoder.load_state_dict(decoder_state, strict=False)
    
    encoder.eval()
    decoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    
    print(f"✓ VAE loaded successfully!")
    print(f"  Latent dimension: {vae_config.latent_dim}")
    
    return encoder, decoder, vae_config


# Diffusion Schedule
class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='linear'):
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'cosine':
            self.alphas_cumprod = self._cosine_beta_schedule(timesteps)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            self.betas = 1 - (self.alphas_cumprod / self.alphas_cumprod_prev)
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
            self.alphas = 1.0 - self.betas
        else:
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return alphas_cumprod[1:]


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# Diffusion Model Components
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class LatentResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim)
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GroupNorm(min(8, out_dim), out_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GroupNorm(min(8, out_dim), out_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x, time_emb):
        h = self.block1(x)
        time_proj = self.time_mlp(time_emb)
        h = h + time_proj
        h = self.block2(h)
        return h + self.residual_proj(x)


class LatentUNet(nn.Module):
    def __init__(self, latent_dim=128, time_dim=256, hidden_dims=[256, 512, 512], dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.dims = [latent_dim] + [hidden_dims[0]] + hidden_dims
        
        self.input_proj = nn.Linear(self.dims[0], self.dims[1])
        
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.encoder_blocks.append(
                LatentResidualBlock(self.dims[i+1], self.dims[i+2], time_dim, dropout)
            )
        
        mid_dim = self.dims[-1]
        self.middle_block1 = LatentResidualBlock(mid_dim, mid_dim, time_dim, dropout)
        self.middle_block2 = LatentResidualBlock(mid_dim, mid_dim, time_dim, dropout)
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            curr_dim = self.dims[-(i+1)]
            skip_dim = self.dims[-(i+2)]
            in_dim = curr_dim + curr_dim
            out_dim = skip_dim
            
            self.decoder_blocks.append(
                LatentResidualBlock(in_dim, out_dim, time_dim, dropout)
            )
        
        self.output_proj = nn.Sequential(
            nn.Linear(self.dims[1] * 2, self.dims[1]),
            nn.SiLU(),
            nn.Linear(self.dims[1], self.dims[0])
        )
    
    def forward(self, x, t):
        time_emb = self.time_embed(t)
        h = self.input_proj(x)
        
        skips = [h]
        for block in self.encoder_blocks:
            h = block(h, time_emb)
            skips.append(h)
        
        h = self.middle_block1(h, time_emb)
        h = self.middle_block2(h, time_emb)
        
        for i, block in enumerate(self.decoder_blocks):
            skip = skips[-(i+1)]
            h = torch.cat([h, skip], dim=-1)
            h = block(h, time_emb)
        
        h = torch.cat([h, skips[0]], dim=-1)
        h = self.output_proj(h)
        
        return h


# Dataset
class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
            self.image_paths.extend(list(self.image_dir.glob(ext.upper())))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


# Training
class EarlyStopper:
    def __init__(self, patience=30, min_delta=1e-5, oscillation_window=10, oscillation_threshold=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.oscillation_window = oscillation_window
        self.oscillation_threshold = oscillation_threshold
        self.counter = 0
        self.best_loss = float('inf')
        self.recent_losses = deque(maxlen=oscillation_window)
    
    def __call__(self, loss):
        self.recent_losses.append(loss)
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
            is_best = False
        
        if len(self.recent_losses) == self.oscillation_window:
            std = np.std(self.recent_losses)
            if std < self.oscillation_threshold:
                return True, "oscillation_detected", is_best
        
        if self.counter >= self.patience:
            return True, "patience_exceeded", is_best
        
        return False, None, is_best


def train_latent_diffusion(
    vae_checkpoint_path,
    image_dir,
    output_dir='/content/MO433/Trabalho Final/main/corel_diffusion_model',
    num_epochs=200,
    batch_size=16,
    grad_accumulation_steps=1,
    learning_rate=1e-4,
    image_size=256,
    gpu_id=None,
    use_ema=True,
    ema_decay=0.995,
    schedule_type='linear',
    use_mixed_precision=True,
    resume_checkpoint=None,
    use_lr_scheduler=True,
    lr_scheduler='cosine_warm_restarts',
    lr_min=1e-5,
    early_stopping_patience=30,
    early_stopping_min_delta=1e-5,
    oscillation_window=10,
    oscillation_threshold=0.0005
):
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("LATENT DIFFUSION TRAINING - COREL DATASET")
    print("="*80)
    print(f"Device: {device}")
    print(f"Image size: {image_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Schedule: {schedule_type}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"EMA: {use_ema}")
    print("="*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    vae_encoder, vae_decoder, vae_config = load_vae_components(vae_checkpoint_path, device)
    latent_dim = vae_config.latent_dim
    
    dataset = ImageDataset(image_dir, image_size)
    print(f"✓ Dataset: {len(dataset)} images\n")
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    model = LatentUNet(
        latent_dim=latent_dim,
        time_dim=256,
        hidden_dims=[256, 512, 512],
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {num_params:,}\n")
    
    schedule = DiffusionSchedule(timesteps=1000, schedule_type=schedule_type)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    if use_lr_scheduler:
        if lr_scheduler == 'cosine_warm_restarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=lr_min
            )
        elif lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=lr_min
            )
        elif lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, min_lr=lr_min
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    scaler = GradScaler() if use_mixed_precision else None
    
    early_stopper = EarlyStopper(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        oscillation_window=oscillation_window,
        oscillation_threshold=oscillation_threshold
    )
    
    class EMA:
        def __init__(self, model, decay):
            self.model = model
            self.decay = decay
            self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        def update(self):
            for name, param in self.model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        
        def apply_shadow(self):
            self.backup = {name: param.clone() for name, param in self.model.named_parameters()}
            for name, param in self.model.named_parameters():
                param.data.copy_(self.shadow[name])
        
        def restore(self):
            for name, param in self.model.named_parameters():
                param.data.copy_(self.backup[name])
    
    ema = EMA(model, ema_decay) if use_ema else None
    
    start_epoch = 0
    loss_history = []
    best_loss = float('inf')
    
    if resume_checkpoint:
        print(f"Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if ema and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
        print(f"✓ Resumed from epoch {start_epoch}\n")
    
    print("="*80)
    print("Training...")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            
            with torch.no_grad():
                mu, logvar = vae_encoder(images)
                latents = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, schedule.timesteps, (latents.shape[0],), device=device).long()
            
            sqrt_alpha_cumprod_t = extract(schedule.sqrt_alphas_cumprod, timesteps, latents.shape)
            sqrt_one_minus_alpha_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, timesteps, latents.shape)
            
            noisy_latents = sqrt_alpha_cumprod_t * latents + sqrt_one_minus_alpha_cumprod_t * noise
            
            if use_mixed_precision:
                with autocast():
                    predicted_noise = model(noisy_latents, timesteps)
                    loss = F.mse_loss(predicted_noise, noise)
                    loss = loss / grad_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if ema:
                        ema.update()
            else:
                predicted_noise = model(noisy_latents, timesteps)
                loss = F.mse_loss(predicted_noise, noise)
                loss = loss / grad_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if ema:
                        ema.update()
            
            epoch_loss += loss.item() * grad_accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item() * grad_accumulation_steps:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        if scheduler:
            if lr_scheduler == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  LR:   {current_lr:.6f}")
        
        should_stop, stop_reason, is_best = early_stopper(avg_loss)
        
        if is_best:
            print(f"✨ New best loss: {avg_loss:.4f}")
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
                'schedule_type': schedule_type,
            }
            if ema:
                checkpoint['ema_shadow'] = ema.shadow
            
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
        
        if should_stop:
            print(f"\n⚠️  Early stopping: {stop_reason}")
            print(f"Best loss: {best_loss:.4f}")
            break
        
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
                'schedule_type': schedule_type,
            }
            if ema:
                checkpoint['ema_shadow'] = ema.shadow
            
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    
    print(f"\n✅ Training complete! Models saved to {output_dir}")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Latent Diffusion for Corel")
    parser.add_argument('--vae-checkpoint', type=str, required=True)
    parser.add_argument('--image-dir', type=str, default='/content/MO433/Trabalho Final/main/data/corel')
    parser.add_argument('--output-dir', type=str, default='/content/MO433/Trabalho Final/main/corel_diffusion_model')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    train_latent_diffusion(
        vae_checkpoint_path=args.vae_checkpoint,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        resume_checkpoint=args.resume
    )
