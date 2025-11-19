"""
Inference Script for Latent Stable Diffusion Model
===================================================
Generate new images using trained latent diffusion model + VAE decoder

Usage:
    python generate_samples.py --checkpoint <path> --vae-checkpoint <path> --output <path> --num-images 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid


# ============================================================================
# VAE COMPONENTS (from training code)
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


def load_vae_decoder(vae_checkpoint_path, device):
    """Load pre-trained VAE decoder"""
    print(f"Loading VAE decoder from: {vae_checkpoint_path}")
    
    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    vae_config = VAEConfig()
    
    decoder = Decoder(vae_config).to(device)
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    
    # Extract decoder weights
    decoder_state = {}
    for k, v in state_dict.items():
        if k.startswith('decoder.'):
            new_key = k[8:]  # Remove 'decoder.' prefix
            decoder_state[new_key] = v
    
    decoder.load_state_dict(decoder_state, strict=False)
    
    # Set to eval mode
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    
    print(f"✓ VAE decoder loaded successfully!")
    return decoder, vae_config


# ============================================================================
# DIFFUSION COMPONENTS
# ============================================================================

class DiffusionSchedule:
    """Manages the noise schedule"""
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='linear'):
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'cosine':
            self.alphas_cumprod = self._cosine_beta_schedule(timesteps)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            self.betas = 1 - (self.alphas_cumprod / self.alphas_cumprod_prev)
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
            self.alphas = 1.0 - self.betas
        else:  # linear
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
    """Residual block for 1D latent vectors"""
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
    """U-Net architecture adapted for 1D latent vectors"""
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
        self.dims = [latent_dim] + [hidden_dims[0]] + hidden_dims
        
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
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            curr_dim = self.dims[-(i+1)]
            skip_dim = self.dims[-(i+2)]
            in_dim = curr_dim + curr_dim
            out_dim = skip_dim
            
            self.decoder_blocks.append(
                LatentResidualBlock(in_dim, out_dim, time_dim, dropout)
            )
        
        # Output projection
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
    """Load trained diffusion model"""
    print(f"Loading diffusion model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get schedule type from checkpoint
    schedule_type = checkpoint.get('schedule_type', 'linear')
    
    # Initialize model
    model = LatentUNet(
        latent_dim=latent_dim,
        time_dim=256,
        hidden_dims=[256, 512, 512],
        dropout=0.1
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Diffusion model loaded successfully!")
    print(f"  Schedule: {schedule_type}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model, schedule_type


# ============================================================================
# SAMPLING
# ============================================================================

@torch.no_grad()
def sample_latents(model, schedule, num_samples=4, latent_dim=128, device='cuda', 
                   show_progress=True, ddim_steps=None):
    """
    Generate latent vectors using DDPM or DDIM sampling.
    
    Args:
        model: Trained diffusion model
        schedule: DiffusionSchedule object
        num_samples: Number of samples to generate
        latent_dim: Dimension of latent space
        device: Device to run on
        show_progress: Show progress bar
        ddim_steps: If not None, use DDIM with this many steps (faster)
    """
    model.eval()
    
    # Start from random noise in latent space
    z = torch.randn(num_samples, latent_dim, device=device)
    
    # Determine timesteps
    if ddim_steps is not None:
        # DDIM: use fewer steps for faster sampling
        timesteps = np.linspace(0, schedule.timesteps - 1, ddim_steps).astype(int)[::-1]
        use_ddim = True
    else:
        # DDPM: use all timesteps
        timesteps = list(range(schedule.timesteps))[::-1]
        use_ddim = False
    
    iterator = tqdm(timesteps, desc="Sampling latents") if show_progress else timesteps
    
    for t in iterator:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(z, t_batch)
        
        if use_ddim:
            # DDIM sampling (faster, deterministic)
            alpha_t = extract(schedule.alphas_cumprod, t_batch, z.shape)
            alpha_t_prev = extract(schedule.alphas_cumprod, 
                                  torch.full_like(t_batch, max(0, t - schedule.timesteps // ddim_steps)), 
                                  z.shape)
            
            sigma = 0.0  # Set to 0 for deterministic, or small value for stochastic
            
            pred_x0 = (z - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma**2) * predicted_noise
            
            if t > 0:
                noise = torch.randn_like(z) if sigma > 0 else 0
                z = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                z = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        else:
            # DDPM sampling (slower, higher quality)
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
    """
    Generate images using diffusion model + VAE decoder
    
    Args:
        diffusion_model: Trained latent diffusion model
        vae_decoder: Trained VAE decoder
        schedule: DiffusionSchedule object
        device: Device to run on
        num_images: Number of images to generate
        latent_dim: Dimension of latent space
        ddim_steps: If not None, use DDIM for faster sampling
        seed: Random seed for reproducibility
    
    Returns:
        Tensor of generated images [N, 3, H, W] in range [0, 1]
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Generate latents using diffusion
    print(f"Generating {num_images} images...")
    if ddim_steps:
        print(f"Using DDIM with {ddim_steps} steps (faster)")
    else:
        print(f"Using DDPM with {schedule.timesteps} steps (higher quality)")
    
    latents = sample_latents(
        diffusion_model, schedule, 
        num_samples=num_images, 
        latent_dim=latent_dim, 
        device=device,
        ddim_steps=ddim_steps
    )
    
    # Decode latents to images using VAE decoder
    print("Decoding latents to images...")
    vae_decoder.eval()
    images = vae_decoder(latents)
    
    # Normalize to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    return images


def save_images(images, output_path, nrow=4):
    """Save images to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as grid
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    save_image(grid, output_path)
    
    print(f"✓ Saved images to: {output_path}")
    
    # Also save individual images if requested
    if len(images) <= 16:  # Only save individuals for small batches
        individual_dir = output_path.parent / f"{output_path.stem}_individual"
        individual_dir.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            save_image(img, individual_dir / f"image_{i:03d}.png")
        print(f"✓ Saved individual images to: {individual_dir}")


def display_images(images, nrow=4, figsize=(12, 12)):
    """Display images using matplotlib"""
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.cpu().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=figsize)
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate images using trained latent diffusion model")
    
    # Required arguments
    parser.add_argument('--diffusion-checkpoint', type=str, required=True,
                       help='Path to trained diffusion model checkpoint')
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                       help='Path to trained VAE checkpoint')
    
    # Generation parameters
    parser.add_argument('--num-images', type=int, default=16,
                       help='Number of images to generate')
    parser.add_argument('--output', type=str, default='generated_images.png',
                       help='Output path for generated images')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Sampling parameters
    parser.add_argument('--ddim-steps', type=int, default=None,
                       help='Use DDIM sampling with N steps (faster, e.g., 50). If None, uses DDPM (slower, 1000 steps)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for generation (reduce if OOM)')
    
    # Display options
    parser.add_argument('--nrow', type=int, default=4,
                       help='Number of images per row in grid')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display images (only save)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    print("="*80)
    print("LATENT DIFFUSION IMAGE GENERATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Number of images: {args.num_images}")
    print(f"Output: {args.output}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("="*80 + "\n")
    
    # Load models
    vae_decoder, vae_config = load_vae_decoder(args.vae_checkpoint, device)
    diffusion_model, schedule_type = load_diffusion_model(
        args.diffusion_checkpoint, 
        latent_dim=vae_config.latent_dim, 
        device=device
    )
    
    # Create diffusion schedule
    schedule = DiffusionSchedule(timesteps=1000, schedule_type=schedule_type)
    
    print()
    
    # Generate images in batches
    all_images = []
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        batch_size = min(args.batch_size, args.num_images - len(all_images))
        
        if num_batches > 1:
            print(f"\nBatch {batch_idx + 1}/{num_batches}")
        
        images = generate_images(
            diffusion_model=diffusion_model,
            vae_decoder=vae_decoder,
            schedule=schedule,
            device=device,
            num_images=batch_size,
            latent_dim=vae_config.latent_dim,
            ddim_steps=args.ddim_steps,
            seed=args.seed + batch_idx if args.seed is not None else None
        )
        
        all_images.append(images)
    
    # Concatenate all batches
    all_images = torch.cat(all_images, dim=0)
    
    print()
    
    # Save images
    save_images(all_images, args.output, nrow=args.nrow)
    
    # Display images
    if not args.no_display:
        try:
            display_images(all_images, nrow=args.nrow)
        except:
            print("(Could not display images - matplotlib may not be configured)")
    
    print("\n" + "="*80)
    print("✅ GENERATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
