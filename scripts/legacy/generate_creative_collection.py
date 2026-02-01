#!/usr/bin/env python3
"""Generate a creative collection with unique themes - different from generate_diverse_collection.py"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_artist.main import AIArtist
from src.ai_artist.utils.config import load_config
from src.ai_artist.utils.logging import configure_logging

configure_logging()

# Completely different prompts from the diverse collection script
prompts = [
    # Cosmic & Space
    "spiral galaxy with vibrant nebula, deep space, cosmic dust, astronomy photography, 4k",
    "astronaut floating in space, earth reflection in helmet, stars, cinematic sci-fi",
    "binary star system, stellar corona, solar flares, space telescope image, dramatic",
    "alien planet surface, two moons in sky, strange rock formations, sci-fi landscape",
    # Mythology & Fantasy Creatures
    "phoenix rising from flames, glowing embers, mythical bird, epic fantasy art, majestic",
    "dragon perched on castle ruins, detailed scales, moonlight, dark fantasy illustration",
    "mermaid swimming in bioluminescent ocean, underwater, magical, ethereal glow",
    "unicorn in enchanted forest, magical sparkles, moonbeams, fantasy storybook art",
    # Retro & Vintage Aesthetics
    "1980s neon city, retrowave sunset, palm trees silhouette, vaporwave aesthetic, gradient sky",
    "vintage arcade machine, pixel art screen, dim lights, 80s nostalgia, retro gaming",
    "1950s diner interior, chrome details, checkered floor, jukebox, americana photography",
    "old film camera collection, leather case, vintage lenses, warm lighting, nostalgic",
    # Surreal & Dreamlike
    "floating islands connected by waterfalls, impossible physics, clouds below, surreal fantasy",
    "door standing alone in desert, no walls, mysterious, minimalist surrealism, sunset",
    "giant teacup with tiny city inside, whimsical, alice in wonderland style, fantasy",
    "infinite staircase in clouds, MC escher inspired, impossible architecture, mind-bending",
    # Seasons & Nature
    "autumn forest trail, red and gold leaves covering ground, soft fog, peaceful morning",
    "cherry blossom avenue, pink petals falling, spring sunrise, soft bokeh, romantic scene",
    "frozen waterfall in winter, icicles, blue ice, snow-covered rocks, cold beauty",
    "sunflower field at golden hour, summer sunset, warm light, pastoral landscape, serene",
    # Portraits & Characters
    "cyberpunk woman with neon tattoos, city lights reflected, rain, futuristic portrait",
    "wise old wizard reading ancient book, candlelight, magical study room, fantasy character",
    "ballerina mid-leap, flowing dress, stage lights, graceful motion, performance art",
    "viking warrior with war paint, fierce expression, dramatic lighting, historical fantasy",
    # Macro Photography
    "dewdrop on spider web, backlit by sunrise, macro detail, bokeh background, nature",
    "butterfly wing scales extreme closeup, iridescent colors, scientific macro photography",
    "snowflake crystal on dark fabric, macro detail, geometric patterns, winter beauty",
    "soap bubble surface, rainbow colors, macro abstract, delicate sphere, artistic",
    # Minimalist Compositions
    "single red umbrella on empty beach, grey sky, minimalist, lonely, artistic photography",
    "solitary tree on rolling hill, negative space, fog, minimalist landscape, zen",
    "geometric shapes, flat colors, bauhaus design, modern minimalism, clean lines",
    "sand ripples pattern, desert, beige tones, minimalist nature, abstract texture",
    # Food & Culinary Art
    "gourmet chocolate dessert, gold leaf decoration, restaurant plating, elegant food photography",
    "colorful macarons arranged on marble, pastel colors, french patisserie, overhead shot",
    "ramen bowl with chopsticks, steam rising, japanese cuisine, cozy food photography",
    "fresh berries in vintage bowl, rustic wooden table, natural light, farm fresh",
    # Weather Phenomena
    "lightning striking over ocean, storm clouds, long exposure, dramatic weather, powerful",
    "morning fog rolling through mountain valley, layers of hills, moody atmosphere, landscape",
    "double rainbow over countryside, rain clearing, hopeful scene, natural beauty",
    "tornado forming in wheat field, ominous sky, storm chaser view, midwest weather",
    # Cultural Landmarks
    "japanese torii gate in water, red shrine, sunset reflection, traditional architecture",
    "moroccan lanterns in souk market, colorful glass, hanging lights, night scene",
    "indian palace courtyard, ornate arches, reflection pool, warm evening light, heritage",
    "norwegian fjord with village, colorful houses, mountains, boats, scandinavian landscape",
    # Experimental Art
    "double exposure portrait merging face with forest, artistic photography, creative blend",
    "light painting spirals, long exposure, dark background, colorful trails, abstract art",
    "infrared landscape, pink trees, surreal colors, false color photography, otherworldly",
    "kaleidoscope effect, symmetrical patterns, vibrant colors, psychedelic art, mandala",
    # Night Scenes
    "city skyline at blue hour, lights turning on, twilight, urban photography, modern",
    "camping under stars, milky way visible, tent with warm glow, wilderness night",
    "bioluminescent beach waves, glowing plankton, night photography, magical natural phenomenon",
    "northern lights over log cabin, aurora borealis, winter night, green and purple sky",
]


async def main():
    """Generate creative artwork collection."""
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

    print("\n🎨 AI Artist - Creative Collection Generator")
    print(f"   Device: {artist.generator.device}")
    print(f"   Model: {artist.generator.model_id}")
    print(f"   Total artworks: {len(prompts)}")
    print("   Theme: Cosmic, Mythical, Retro, Surreal & More!")

    # Generate images
    print(f"\n🖼️  Generating {len(prompts)} creative artworks...\n")
    success_count = 0
    failed_prompts = []

    for i, prompt in enumerate(prompts, 1):
        try:
            # Show progress
            category = prompt.split(",")[0][:50]
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
                    "seed": "random",
                    "collection": "creative",
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
                print("   ❌ No images generated\n")
                failed_prompts.append((i, prompt))

        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            failed_prompts.append((i, prompt))
            continue

    # Summary
    print(f"\n{'=' * 60}")
    print("✨ Generation Complete!")
    print(f"{'=' * 60}")
    print(f"✅ Successful: {success_count}/{len(prompts)}")
    print(f"❌ Failed: {len(failed_prompts)}")

    if failed_prompts:
        print("\n⚠️  Failed prompts:")
        for idx, prompt in failed_prompts:
            print(f"   [{idx}] {prompt[:60]}...")

    print("\n🌐 View gallery: http://localhost:8000")
    print("📁 Images saved to: gallery/")


if __name__ == "__main__":
    asyncio.run(main())
