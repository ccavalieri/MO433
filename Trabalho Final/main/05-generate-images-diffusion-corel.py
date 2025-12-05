#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Synthetic Images using VAE + Diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid


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


def load_vae_decoder(vae_checkpoint_path, device):
    print(f"Loading VAE decoder from: {vae_checkpoint_path}")
    
    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    vae_config = VAEConfig()
    
    decoder = Decoder(vae_config).to(device)
    
    state_dict = checkpoint['model_state_dict']
    
    decoder_state = {}
    for k, v in state_dict.items():
        if k.startswith('decoder.'):
            new_key = k[8:]
            decoder_state[new_key] = v
    
    decoder.load_state_dict(decoder_state, strict=False)
    
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    
    return decoder, vae_config


# Diffusion Components
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


def load_diffusion_model(checkpoint_path, latent_dim=128, device='cuda'):
    print(f"Loading diffusion model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    schedule_type = checkpoint.get('schedule_type', 'linear')
    
    model = LatentUNet(
        latent_dim=latent_dim,
        time_dim=256,
        hidden_dims=[256, 512, 512],
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, schedule_type


# Sampling
@torch.no_grad()
def sample_latents(model, schedule, num_samples=4, latent_dim=128, device='cuda', 
                   show_progress=True, ddim_steps=None):
    model.eval()
    
    z = torch.randn(num_samples, latent_dim, device=device)
    
    if ddim_steps is not None:
        timesteps = np.linspace(0, schedule.timesteps - 1, ddim_steps).astype(int)[::-1]
        use_ddim = True
    else:
        timesteps = list(range(schedule.timesteps))[::-1]
        use_ddim = False
    
    iterator = tqdm(timesteps, desc="Sampling latents") if show_progress else timesteps
    
    for t in iterator:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        predicted_noise = model(z, t_batch)
        
        if use_ddim:
            alpha_t = extract(schedule.alphas_cumprod, t_batch, z.shape)
            alpha_t_prev = extract(schedule.alphas_cumprod, 
                                  torch.full_like(t_batch, max(0, t - schedule.timesteps // ddim_steps)), 
                                  z.shape)
            
            sigma = 0.0
            
            pred_x0 = (z - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma**2) * predicted_noise
            
            if t > 0:
                noise = torch.randn_like(z) if sigma > 0 else 0
                z = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                z = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        else:
            alpha_t = extract(schedule.alphas, t_batch, z.shape)
            beta_t = extract(schedule.betas, t_batch, z.shape)
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
            sqrt_one_minus_alpha_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t_batch, z.shape)
            
            posterior_mean = sqrt_recip_alpha_t * (z - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(z)
                posterior_variance_t = extract(schedule.posterior_variance, t_batch, z.shape)
                z = posterior_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                z = posterior_mean
    
    return z


@torch.no_grad()
def generate_images(diffusion_model, vae_decoder, schedule, device='cuda',
                   num_images=16, latent_dim=128, ddim_steps=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    print(f"Generating {num_images} images")
    
    latents = sample_latents(
        diffusion_model, schedule, 
        num_samples=num_images, 
        latent_dim=latent_dim, 
        device=device,
        ddim_steps=ddim_steps
    )
    
    vae_decoder.eval()
    images = vae_decoder(latents)
    
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    return images


# Class info
CLASS_INFO = {
    "0001": "british_guards",
    "0002": "locomotives",
    "0003": "desserts",
    "0004": "salads",
    "0005": "snow",
    "0006": "sunset"
}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate images using VAE + Diffusion")
    
    parser.add_argument('--diffusion-checkpoint', type=str, required=True)
    parser.add_argument('--vae-checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='/content/MO433/Trabalho Final/main/generated_images_diffusion_corel')
    parser.add_argument('--num-images-per-class', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ddim-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print("IMAGE GENERATION - VAE + DIFFUSION")
    print(f"Device: {device}")
    print(f"Images per class: {args.num_images_per_class}")
    print(f"Total images: {args.num_images_per_class * len(CLASS_INFO)}")
    print(f"Output: {args.output_dir}")
    
    vae_decoder, vae_config = load_vae_decoder(args.vae_checkpoint, device)
    diffusion_model, schedule_type = load_diffusion_model(
        args.diffusion_checkpoint, 
        latent_dim=vae_config.latent_dim, 
        device=device
    )
    
    schedule = DiffusionSchedule(timesteps=1000, schedule_type=schedule_type)
    
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_generated = 0
    
    for class_id in sorted(CLASS_INFO.keys()):
        class_name = CLASS_INFO[class_id]
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)            
        
        num_batches = (args.num_images_per_class + args.batch_size - 1) // args.batch_size
        
        all_images = []
        
        for batch_idx in tqdm(range(num_batches), desc=f"Class {class_id}"):
            batch_size = min(args.batch_size, args.num_images_per_class - len(all_images))
            
            images = generate_images(
                diffusion_model=diffusion_model,
                vae_decoder=vae_decoder,
                schedule=schedule,
                device=device,
                num_images=batch_size,
                latent_dim=vae_config.latent_dim,
                ddim_steps=args.ddim_steps,
                seed=args.seed + batch_idx + int(class_id) * 1000
            )
            
            for i, img in enumerate(images):
                img_idx = len(all_images) + i + 1
                img_filename = f"{class_id}_{img_idx:04d}_diffusion.png"
                save_image(img, class_dir / img_filename)
            
            all_images.append(images)
            
            torch.cuda.empty_cache()
        
        all_images = torch.cat(all_images, dim=0)
        
        grid = make_grid(all_images[:16], nrow=4, padding=2, normalize=False)
        save_image(grid, class_dir / f"{class_name}_grid.png")
        
        total_generated += len(all_images)
        print(f"Generated {len(all_images)} images for {class_name}")
    
    print("Generation complete")
    print(f"Total images generated: {total_generated}")
    print(f"Output directory: {args.output_dir}")
    print("\nImages per class:")
    for class_id in sorted(CLASS_INFO.keys()):
        class_name = CLASS_INFO[class_id]
        print(f"  {class_id} ({class_name}): {args.num_images_per_class} images")


if __name__ == "__main__":
    main()
