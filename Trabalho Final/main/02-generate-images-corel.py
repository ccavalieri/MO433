#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Synthetic Images for Corel Dataset
"""

from diffusers import StableDiffusionPipeline
import torch
from diffusers.utils import make_image_grid
from diffusers import EulerDiscreteScheduler
from datetime import datetime
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
LORA_DIR = "./corel_lora_model"
OUTPUT_DIR = "./generated_images_corel"
RESOLUTION = 256
NUM_IMAGES_PER_CLASS = 50
SEED_START = 42
GUIDANCE_SCALE = 7.5

# Class definitions
CLASS_INFO = {
    "0001": {
        "name": "british_guards",
        "prompt": "british royal guards in red uniforms, ceremonial military parade, high quality photo",
        "negative": "low quality, blur, watermark, text, distorted"
    },
    "0002": {
        "name": "locomotives",
        "prompt": "steam locomotive train, vintage railway engine, high quality photo",
        "negative": "low quality, blur, watermark, text, distorted"
    },
    "0003": {
        "name": "desserts",
        "prompt": "desserts and sweets, pastries and confectionery, high quality photo",
        "negative": "low quality, blur, watermark, text, distorted"
    },
    "0004": {
        "name": "salads",
        "prompt": "fresh salad with vegetables and fruits, healthy food, high quality photo",
        "negative": "low quality, blur, watermark, text, distorted"
    },
    "0005": {
        "name": "snow",
        "prompt": "winter snow scene, snowy landscape, high quality photo",
        "negative": "low quality, blur, watermark, text, distorted"
    },
    "0006": {
        "name": "sunset",
        "prompt": "sunset over water, orange sky at dusk, high quality photo",
        "negative": "low quality, blur, watermark, text, distorted"
    }
}

def find_latest_lora(lora_dir):
    """Find the most recent LoRA checkpoint"""
    lora_path = Path(lora_dir)
    lora_files = list(lora_path.glob("*.safetensors"))
    
    if not lora_files:
        raise FileNotFoundError(f"No LoRA files found in {lora_dir}")
    
    # Sort by modification time
    latest_lora = max(lora_files, key=lambda x: x.stat().st_mtime)
    return latest_lora.name

def setup_pipeline(lora_dir, lora_name, device="cuda:0"):
    """Setup Stable Diffusion pipeline with LoRA"""
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    # Memory optimizations
    pipe.enable_attention_slicing(slice_size=1)
    pipe.enable_vae_tiling()
    
    # Load LoRA
    print(f"\nLoading LoRA: {lora_name}")
    pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=lora_dir,
        weight_name=lora_name,
        adapter_name="corel_lora"
    )
    pipe.set_adapters(["corel_lora"], adapter_weights=[1.0])
    
    # Set scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    return pipe

def generate_class_images(pipe, class_id, class_info, output_dir, num_images, resolution, seed_start, guidance_scale, device):
    """Generate images for a specific class"""
    class_name = class_info["name"]
    prompt = class_info["prompt"]
    negative_prompt = class_info["negative"]
    
    # Create class directory
    class_dir = Path(output_dir) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating {num_images} images for class {class_id} ({class_name})")
    print(f"Prompt: {prompt}")
    print(f"Output: {class_dir}")
    
    images = []
    
    for i in tqdm(range(num_images), desc=f"Class {class_id}"):
        seed = seed_start + i
        
        # Generate single image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            generator=torch.Generator(device).manual_seed(seed),
            width=resolution,
            height=resolution,
            guidance_scale=guidance_scale
        ).images[0]
        
        # Save individual image
        img_filename = f"{class_id}_{i+1:04d}_synthetic.png"
        img_path = class_dir / img_filename
        image.save(img_path)
        
        images.append(image)
        
        # Clear cache periodically
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Create grid for this class
    grid = make_image_grid(images[:16], cols=4, rows=4)  # First 16 images
    grid_path = class_dir / f"{class_name}_grid.png"
    grid.save(grid_path)
    
    return len(images)

def main():
    # Find latest LoRA
    try:
        lora_name = find_latest_lora(LORA_DIR)
        print(f"Found LoRA: {lora_name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Setup output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup pipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = setup_pipeline(LORA_DIR, lora_name, device)
    
    # Generate images for each class
    total_generated = 0
    for class_id in sorted(CLASS_INFO.keys()):
        class_info = CLASS_INFO[class_id]
        
        num_generated = generate_class_images(
            pipe=pipe,
            class_id=class_id,
            class_info=class_info,
            output_dir=OUTPUT_DIR,
            num_images=NUM_IMAGES_PER_CLASS,
            resolution=RESOLUTION,
            seed_start=SEED_START,
            guidance_scale=GUIDANCE_SCALE,
            device=device
        )
        
        total_generated += num_generated
    
    # Cleanup
    pipe.to("cpu")
    torch.cuda.empty_cache()
    
    print(f"Total images generated: {total_generated}")

if __name__ == "__main__":
    main()
