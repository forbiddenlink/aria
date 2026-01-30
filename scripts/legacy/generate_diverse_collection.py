#!/usr/bin/env python3
"""Generate a diverse collection of artwork across multiple styles and themes."""

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_artist.main import AIArtist
from src.ai_artist.utils.logging import configure_logging
from src.ai_artist.utils.config import load_config

configure_logging()

# Highly diverse prompts covering multiple styles, subjects, and aesthetics
prompts = [
    # Nature & Landscapes
    "misty forest at dawn, ethereal light rays, mystical atmosphere, highly detailed",
    "underwater coral reef, colorful fish, sunlight filtering through water, photorealistic",
    "northern lights over snowy mountains, starry sky, purple and green aurora, dramatic",
    "desert sand dunes at sunset, golden hour, long shadows, minimalist composition",
    
    # Urban & Architecture  
    "tokyo street at night, neon signs, rain reflections, cyberpunk aesthetic, cinematic",
    "ancient library interior, wooden shelves, warm candlelight, gothic architecture, atmospheric",
    "modern glass skyscraper, blue sky reflection, minimalist, architectural photography",
    "cobblestone european alley, flower boxes, cafe tables, warm afternoon light, romantic",
    
    # Abstract & Artistic
    "fluid art, swirling colors, gold and turquoise, marble texture, abstract expressionism",
    "geometric mandala, intricate patterns, vibrant gradients, symmetrical, digital art",
    "watercolor landscape, soft brushstrokes, pastel colors, dreamy, impressionist style",
    "fractal patterns, psychedelic colors, sacred geometry, infinite depth, visionary art",
    
    # Fantasy & Sci-Fi
    "steampunk airship, brass machinery, clouds, Victorian era, fantasy illustration",
    "alien planet landscape, two moons, strange plants, purple sky, science fiction",
    "crystal cave, glowing minerals, underground lake, magical, fantasy art",
    "space station interior, futuristic design, holographic displays, sci-fi concept art",
    
    # Still Life & Objects
    "vintage typewriter, old books, coffee cup, wooden desk, warm window light, nostalgic",
    "fresh fruit bowl, dramatic lighting, dark background, dutch masters style, oil painting",
    "origami birds, pastel paper, soft shadows, minimalist japanese aesthetic, serene",
    "antique pocket watch, mechanical gears visible, macro photography, steampunk details",
    
    # Animals & Wildlife
    "peacock displaying feathers, vibrant colors, botanical garden, wildlife photography",
    "arctic fox in snow, winter landscape, blue hour lighting, nature documentary style",
    "hummingbird feeding on flower, frozen motion, macro, vibrant tropical colors",
    "elephant family at watering hole, golden sunset, african savanna, dramatic",
    
    # Seasonal & Weather
    "autumn forest path, falling leaves, warm colors, peaceful, golden hour",
    "cherry blossom tree, pink petals falling, spring morning, soft bokeh, japanese garden",
    "thunderstorm over wheat field, dramatic clouds, lightning, moody atmosphere",
    "winter cabin, snow falling, warm windows, pine trees, cozy evening scene",
    
    # Cultural & Traditional
    "moroccan courtyard, ornate tiles, fountain, arches, warm sunlight, architectural detail",
    "japanese tea ceremony, tatami room, minimalist, soft natural light, cultural",
    "indian spice market, colorful powders, vendors, busy scene, vibrant street photography",
    "scandinavian interior, white walls, natural wood, plants, hygge aesthetic, minimal",
]

async def main():
    """Generate diverse artwork collection."""
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
    
    print(f"\n🎨 AI Artist - Diverse Collection Generator")
    print(f"   Device: {artist.generator.device}")
    print(f"   Model: {artist.generator.model_id}")
    print(f"   Total artworks: {len(prompts)}")
    
    # Generate images
    print(f"\n🖼️  Generating {len(prompts)} diverse artworks...\n")
    success_count = 0
    failed_prompts = []
    
    for i, prompt in enumerate(prompts, 1):
        try:
            # Show progress
            category = prompt.split(',')[0][:50]
            print(f"[{i}/{len(prompts)}] {category}...")
            
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
                print(f"   ✅ Saved: {saved_path.name}\n")
            else:
                print(f"   ❌ No images generated\n")
                failed_prompts.append((i, prompt))
                
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            failed_prompts.append((i, prompt))
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✨ Generation Complete!")
    print(f"{'='*60}")
    print(f"✅ Successful: {success_count}/{len(prompts)}")
    print(f"❌ Failed: {len(failed_prompts)}")
    
    if failed_prompts:
        print(f"\n⚠️  Failed prompts:")
        for idx, prompt in failed_prompts:
            print(f"   [{idx}] {prompt[:60]}...")
    
    print(f"\n🌐 View gallery: http://localhost:8000")
    print(f"📁 Images saved to: gallery/")

if __name__ == "__main__":
    asyncio.run(main())
