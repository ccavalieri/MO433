"""
Stable Diffusion - Latent Space Diffusion Model
================================================
This code trains a diffusion model in the latent space of a pre-trained VAE.
It loads the encoder and decoder from code4-train-vae.py and applies diffusion
to the latent vectors, then uses the decoder to generate final images.

Features:
- Works in VAE latent space (much more efficient)
- Cosine Annealing with Warm Restarts
- Early Stopping with oscillation detection
- Mixed precision training (FP16)
- Gradient accumulation
- EMA for stable generation
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
import warnings
from pathlib import Path


# ============================================================================
# VAE COMPONENTS (from code4-train-vae.py)
# ============================================================================

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


class VAEConfig:
    """Configuration matching code4-train-vae.py"""
    image_size = 128
    image_channels = 3
    latent_dim = 128
    hidden_dims = [64, 128, 256, 512]


def load_vae_components(vae_checkpoint_path, device):
    """Load pre-trained VAE encoder and decoder from code4 checkpoint"""
    print(f"Loading VAE from: {vae_checkpoint_path}")
    
    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    vae_config = VAEConfig()
    
    encoder = Encoder(vae_config).to(device)
    decoder = Decoder(vae_config).to(device)
    
    # Load state dict - the checkpoint saves the full VAE model
    state_dict = checkpoint['model_state_dict']
    
    # Separate encoder and decoder weights based on key patterns
    encoder_state = {}
    decoder_state = {}
    
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            # Remove 'encoder.' prefix: 'encoder.encoder.0.0.weight' -> 'encoder.0.0.weight'
            # or 'encoder.fc_mu.weight' -> 'fc_mu.weight'
            new_key = k[8:]  # Remove 'encoder.' prefix
            encoder_state[new_key] = v
        elif k.startswith('decoder.'):
            # Remove 'decoder.' prefix
            new_key = k[8:]  # Remove 'decoder.' prefix
            decoder_state[new_key] = v
    
    # Load the weights
    missing_enc, unexpected_enc = encoder.load_state_dict(encoder_state, strict=False)
    missing_dec, unexpected_dec = decoder.load_state_dict(decoder_state, strict=False)
    
    if missing_enc:
        print(f"⚠️  Missing encoder keys: {missing_enc[:5]}...")  # Show first 5
    if missing_dec:
        print(f"⚠️  Missing decoder keys: {missing_dec[:5]}...")
    
    # Freeze VAE weights
    encoder.eval()
    decoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    
    print(f"✓ VAE loaded successfully!")
    print(f"  Latent dimension: {vae_config.latent_dim}")
    print(f"  Encoder params loaded: {len(encoder_state)}")
    print(f"  Decoder params loaded: {len(decoder_state)}")
    
    return encoder, decoder, vae_config


# ============================================================================
# DIFFUSION SCHEDULE AND UTILITIES
# ============================================================================

class DiffusionSchedule:
    """Manages the noise schedule - DEFAULT: Linear (proven to work)"""
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='linear'):
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'cosine':
            self.alphas_cumprod = self._cosine_beta_schedule(timesteps)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            self.betas = 1 - (self.alphas_cumprod / self.alphas_cumprod_prev)
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
            self.alphas = 1.0 - self.betas
        else:  # linear (DEFAULT)
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
    """Extract coefficients at specified timesteps."""
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class EMA:
    """Exponential Moving Average"""
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA shadow weights with current model weights (after backprop)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Temporarily replace model weights with EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:
    """Early stopping with loss plateau and oscillation detection."""
    def __init__(self, patience=20, min_delta=1e-5, window_size=10, oscillation_threshold=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.oscillation_threshold = oscillation_threshold
        
        self.best_loss = float('inf')
        self.counter = 0
        self.loss_history = deque(maxlen=window_size)
        self.should_stop = False
        
    def __call__(self, loss):
        self.loss_history.append(loss)
        is_best = False
        reason = None
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            reason = f"Loss plateau detected ({self.patience} epochs without improvement)"
            return True, reason, is_best
        
        if len(self.loss_history) == self.window_size:
            mean_loss = np.mean(self.loss_history)
            std_loss = np.std(self.loss_history)
            
            if std_loss < self.oscillation_threshold and abs(loss - mean_loss) < self.oscillation_threshold:
                self.should_stop = True
                reason = f"Loss oscillating around {mean_loss:.4f} (std={std_loss:.6f})"
                return True, reason, is_best
        
        return False, None, is_best
    
    def get_stats(self):
        if len(self.loss_history) == 0:
            return None
        return {
            'mean': np.mean(self.loss_history),
            'std': np.std(self.loss_history),
            'best': self.best_loss,
            'patience_counter': self.counter
        }


# ============================================================================
# LATENT DIFFUSION MODEL (U-Net for Latent Space)
# ============================================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
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
    """
    Residual block for 1D latent vectors
    Adapted for latent space processing
    """
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
        
        # Residual connection projection if dimensions don't match
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_proj = self.time_mlp(time_emb)
        h = h + time_proj
        
        h = self.block2(h)
        
        # Add residual with projection if needed
        return h + self.residual_proj(x)


class LatentUNet(nn.Module):
    """
    U-Net architecture adapted for 1D latent vectors
    Simplified with explicit dimension tracking
    """
    def __init__(self, latent_dim=128, time_dim=256, hidden_dims=[256, 512, 512], dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Build explicit dimension progression
        # input: 128 -> proj -> 256 -> enc1 -> 512 -> enc2 -> 512 -> enc3 -> 512
        self.dims = [latent_dim] + [hidden_dims[0]] + hidden_dims
        # self.dims = [128, 256, 512, 512, 512]
        
        # Initial projection
        self.input_proj = nn.Linear(self.dims[0], self.dims[1])
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.encoder_blocks.append(
                LatentResidualBlock(self.dims[i+1], self.dims[i+2], time_dim, dropout)
            )
        
        # Middle blocks
        mid_dim = self.dims[-1]
        self.middle_block1 = LatentResidualBlock(mid_dim, mid_dim, time_dim, dropout)
        self.middle_block2 = LatentResidualBlock(mid_dim, mid_dim, time_dim, dropout)
        
        # Decoder blocks (reverse, with skip connections)
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            # Current dimension from previous layer
            curr_dim = self.dims[-(i+1)]
            # Skip connection dimension (matching encoder)
            skip_dim = self.dims[-(i+2)]
            # Input is concatenation
            in_dim = curr_dim + curr_dim  # Both current and skip have same dim
            out_dim = skip_dim
            
            self.decoder_blocks.append(
                LatentResidualBlock(in_dim, out_dim, time_dim, dropout)
            )
        
        # Output projection (with skip from initial projection)
        self.output_proj = nn.Sequential(
            nn.Linear(self.dims[1] * 2, self.dims[1]),
            nn.SiLU(),
            nn.Linear(self.dims[1], self.dims[0])
        )
    
    def forward(self, x, t):
        # Time embedding
        time_emb = self.time_embed(t)
        
        # Initial projection
        h = self.input_proj(x)  # 128 -> 256
        
        # Store skip connections
        skips = [h]  # Start with initial projection
        
        # Encoder
        for block in self.encoder_blocks:
            h = block(h, time_emb)
            skips.append(h)
        # skips = [256, 512, 512, 512]
        
        # Middle
        h = self.middle_block1(h, time_emb)  # 512 -> 512
        h = self.middle_block2(h, time_emb)  # 512 -> 512
        
        # Decoder (use skips in reverse, except the first one)
        for i, block in enumerate(self.decoder_blocks):
            skip = skips[-(i+1)]  # Get corresponding skip in reverse
            h = torch.cat([h, skip], dim=-1)
            h = block(h, time_emb)
        
        # Final output with first skip
        h = torch.cat([h, skips[0]], dim=-1)
        h = self.output_proj(h)
        
        return h


# ============================================================================
# DATASET
# ============================================================================

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=128):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.image_dir.rglob(ext)))
            self.image_paths.extend(list(self.image_dir.rglob(ext.upper())))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
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


# ============================================================================
# TRAINING
# ============================================================================

def train_latent_diffusion(
    vae_checkpoint_path,
    image_dir,
    output_dir='latent_diffusion',
    num_epochs=500,
    batch_size=16,
    grad_accumulation_steps=1,
    learning_rate=1e-4,
    image_size=128,
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
    # Device setup
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Load VAE components
    vae_encoder, vae_decoder, vae_config = load_vae_components(vae_checkpoint_path, device)
    latent_dim = vae_config.latent_dim
    
    # Dataset and dataloader
    dataset = ImageDataset(image_dir, image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize diffusion schedule
    schedule = DiffusionSchedule(timesteps=1000, schedule_type=schedule_type)
    
    # Initialize latent diffusion model
    model = LatentUNet(
        latent_dim=latent_dim,
        time_dim=256,
        hidden_dims=[256, 512, 512],
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Latent Diffusion Model Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = None
    if use_lr_scheduler:
        if lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=lr_min
            )
        elif lr_scheduler == 'cosine_warm_restarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2, eta_min=lr_min
            )
        elif lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, min_lr=lr_min
            )
    
    # EMA
    ema = EMA(model, decay=ema_decay) if use_ema else None
    
    # Early stopping
    early_stopper = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        window_size=oscillation_window,
        oscillation_threshold=oscillation_threshold
    )
    
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    loss_history = []
    
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if ema and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
        print(f"✓ Resumed from epoch {start_epoch}")
    
    print("\n" + "="*80)
    print("TRAINING LATENT DIFFUSION MODEL")
    print("="*80)
    print(f"Dataset size:     {len(dataset)}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Batch size:       {batch_size}")
    print(f"Epochs:           {num_epochs}")
    print(f"Learning rate:    {learning_rate}")
    print(f"Schedule:         {schedule_type}")
    print(f"EMA:              {'Yes' if use_ema else 'No'}")
    print(f"Mixed precision:  {'Yes' if use_mixed_precision else 'No'}")
    print("="*80 + "\n")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, images in enumerate(pbar):
            images = images.to(device)
            
            # Encode images to latent space using VAE encoder
            with torch.no_grad():
                mu, logvar = vae_encoder(images)
                # Sample from posterior for better distribution
                # This helps when VAE has low KL divergence
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                latents = mu + eps * std  # Stochastic encoding
            
            # Sample random timesteps
            t = torch.randint(0, schedule.timesteps, (latents.shape[0],), device=device).long()
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, latents.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, latents.shape)
            noisy_latents = sqrt_alphas_cumprod_t * latents + sqrt_one_minus_alphas_cumprod_t * noise
            
            # Forward pass with mixed precision
            if use_mixed_precision:
                with autocast():
                    predicted_noise = model(noisy_latents, t)
                    loss = F.mse_loss(predicted_noise, noise)
                
                # Backward pass with gradient scaling
                scaler.scale(loss / grad_accumulation_steps).backward()
                
                if (step + 1) % grad_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if ema:
                        ema.update()
            else:
                predicted_noise = model(noisy_latents, t)
                loss = F.mse_loss(predicted_noise, noise)
                
                (loss / grad_accumulation_steps).backward()
                
                if (step + 1) % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if ema:
                        ema.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Update learning rate
        if scheduler:
            if lr_scheduler == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  LR:   {current_lr:.6f}")
        
        # Early stopping check
        should_stop, stop_reason, is_best = early_stopper(avg_loss)
        
        # Save best model
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
        
        # Check if we should stop
        if should_stop:
            print(f"\n⚠️  Early stopping triggered: {stop_reason}")
            print(f"Best loss achieved: {best_loss:.4f}")
            break
        
        # Checkpoint and samples every 10 epochs
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
            
            # Generate samples
            print(f"Generating samples...")
            if ema:
                ema.apply_shadow()
            generate_samples(
                model, vae_decoder, schedule, device,
                os.path.join(output_dir, 'samples', f'epoch_{epoch+1}.png'),
                latent_dim=latent_dim,
                num_images=4
            )
            if ema:
                ema.restore()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
    
    # Plot loss history
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


# ============================================================================
# SAMPLING
# ============================================================================

@torch.no_grad()
def sample_latents(model, schedule, num_samples=4, latent_dim=128, device='cuda'):
    """Generate latent vectors using DDPM sampling."""
    model.eval()
    
    # Start from random noise in latent space
    z = torch.randn(num_samples, latent_dim, device=device)
    
    for t in tqdm(reversed(range(schedule.timesteps)), desc="Sampling latents", total=schedule.timesteps):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(z, t_batch)
        
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


def generate_samples(model, vae_decoder, schedule, device, save_path, latent_dim=128, num_images=4):
    """Generate and save sample images."""
    # Generate latents using diffusion
    latents = sample_latents(model, schedule, num_images, latent_dim, device)
    
    # Decode latents to images using VAE decoder
    vae_decoder.eval()
    with torch.no_grad():
        images = vae_decoder(latents)
    
    # Normalize to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Save images
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))
    if num_images == 1:
        axes = [axes]
    
    for idx, ax in enumerate(axes):
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved samples to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stable Diffusion - Latent Space Diffusion Model")
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                       help='Path to VAE checkpoint from code4-train-vae.py')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--output-dir', type=str, default='latent_diffusion',
                       help='Output directory for models and samples')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--gpu-id', type=int, default=None)
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.995)
    parser.add_argument('--schedule', type=str, default='linear', choices=['cosine', 'linear'])
    parser.add_argument('--no-mixed-precision', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lr-scheduler', type=str, default='cosine_warm_restarts',
                       choices=['cosine', 'cosine_warm_restarts', 'plateau', 'none'])
    parser.add_argument('--lr-min', type=float, default=1e-5)
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--early-stopping-delta', type=float, default=1e-5)
    parser.add_argument('--oscillation-window', type=int, default=10)
    parser.add_argument('--oscillation-threshold', type=float, default=0.0005)
    
    args = parser.parse_args()
    
    train_latent_diffusion(
        vae_checkpoint_path=args.vae_checkpoint,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        image_size=args.image_size,
        gpu_id=args.gpu_id,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        schedule_type=args.schedule,
        use_mixed_precision=not args.no_mixed_precision,
        resume_checkpoint=args.resume,
        use_lr_scheduler=(args.lr_scheduler != 'none'),
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_delta,
        oscillation_window=args.oscillation_window,
        oscillation_threshold=args.oscillation_threshold
    )
