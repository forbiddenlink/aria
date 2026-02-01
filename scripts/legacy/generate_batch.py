#!/usr/bin/env python3
"""Generate multiple diverse artworks for the gallery."""

import asyncio
from pathlib import Path

from src.ai_artist.core.generator import ImageGenerator

# Diverse prompts for interesting artwork
PROMPTS = [
    "a serene mountain landscape at golden hour, misty valleys, photorealistic",
    "abstract geometric patterns in vibrant colors, modern art style",
    "a cozy coffee shop interior with warm lighting, aesthetic, detailed",
    "futuristic cityscape with flying vehicles, neon lights, cyberpunk style",
    "a peaceful zen garden with cherry blossoms and koi pond, Japanese art",
    "vintage retro poster design with bold colors, 1960s style",
    "a majestic lion in the African savanna at sunset, wildlife photography",
    "minimalist landscape with rolling hills and single tree, artistic",
]


async def main():
    config_path = Path("config/config.yaml")
    generator = ImageGenerator(config_path)

    print(f"\n🎨 Generating {len(PROMPTS)} artworks...\n")

    success_count = 0
    for i, prompt in enumerate(PROMPTS, 1):
        try:
            print(f"[{i}/{len(PROMPTS)}] Generating: {prompt[:60]}...")
            result = generator.generate(
                prompt=prompt,
                num_images=1,
                steps=25,
            )
            print(f"✅ Success! Generated {len(result['images'])} image(s)\n")
            success_count += 1
        except Exception as e:
            print(f"❌ Error: {e}\n")
            continue

    print(
        f"\n✨ Complete! Successfully generated {success_count}/{len(PROMPTS)} artworks"
    )


if __name__ == "__main__":
    asyncio.run(main())
