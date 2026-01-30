#!/usr/bin/env python3
"""Generate artwork using the AI Artist system with proper DB integration."""

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_artist.main import AIArtist
from src.ai_artist.utils.logging import configure_logging
from src.ai_artist.utils.config import load_config

configure_logging()

# Diverse prompts for testing
prompts = [
    "a beautiful sunset over mountains, golden hour, dramatic clouds, photorealistic",
    "abstract geometric patterns, vibrant colors, modern art, high contrast",
    "cozy coffee shop interior, warm lighting, bokeh, cinematic",
    "cyberpunk city at night, neon lights, rain, futuristic, detailed",
    "zen garden with cherry blossoms, peaceful, minimalist, japanese aesthetic",
    "vintage car on desert highway, retro, film grain, nostalgic",
    "majestic lion portrait, wildlife photography, golden light, detailed fur",
    "minimalist architecture, clean lines, concrete and glass, modernist",
]

async def main():
    """Generate diverse artwork."""
    # Load config
    config_path = Path("config/config.yaml")
    config = load_config(config_path)
    
    # Initialize AI Artist  
    artist = AIArtist(config)
    
    # Override to use CPU (MPS is too slow)
    import torch
    artist.generator.device = "cpu"
    artist.generator.dtype = torch.float32
    artist.generator.model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load the model
    artist.generator.load_model()
    
    print(f"\n🎨 AI Artist initialized")
    print(f"   Device: {artist.generator.device}")
    print(f"   Model: {artist.generator.model_id}")
    
    # Generate images
    print(f"\n🖼️  Generating {len(prompts)} artworks...\n")
    success_count = 0
    
    for i, prompt in enumerate(prompts, 1):
        try:
            print(f"[{i}/{len(prompts)}] {prompt[:60]}...")
            
            # Generate images using the generator directly
            images = artist.generator.generate(
                prompt=prompt,
                num_images=1,
                num_inference_steps=15,
            )
            
            # Save with gallery manager
            if images:
                metadata = {
                    "prompt": prompt,
                    "model": artist.generator.model_id,
                    "seed": "noseed",
                }
                saved_path = artist.gallery.save_image(
                    image=images[0],
                    prompt=prompt,
                    metadata=metadata,
                    featured=False,
                )
                success_count += 1
                print(f"   ✅ Saved to: {saved_path}\n")
            else:
                print(f"   ❌ No images generated\n")
                
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            continue
    
    print(f"\n✨ Complete! Generated {success_count}/{len(prompts)} artworks")
    print(f"🌐 View gallery: http://localhost:8000")

if __name__ == "__main__":
    asyncio.run(main())
