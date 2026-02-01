#!/usr/bin/env python3
"""Generate random images with wildly different themes and styles.

This script creates a diverse collection of images using random prompts
across various artistic styles, subjects, and aesthetics.
"""

import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_artist.core.generator import ImageGenerator
from src.ai_artist.gallery.manager import GalleryManager
from src.ai_artist.utils.config import load_config
from src.ai_artist.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# Diverse random prompts - completely different from other scripts
RANDOM_PROMPTS = {
    "cosmic": [
        "spiral galaxy with vibrant nebula clouds, deep space, cosmic colors, astronomy photography",
        "black hole event horizon, warped spacetime, accretion disk, scientific visualization",
        "binary star system, stellar corona, solar flares, space telescope image",
        "supernova explosion, shock waves, cosmic dust, deep field photography",
        "planet with rings, multiple moons, gas giant, cinematic space art",
    ],
    "mythology": [
        "phoenix rising from flames, mythical bird, fire and embers, epic fantasy art",
        "dragon perched on mountain peak, scales detail, medieval fantasy, dramatic lighting",
        "mermaid in bioluminescent ocean, underwater scene, magical realism, ethereal",
        "griffon guardian statue, ancient temple ruins, overgrown with vines, mystical",
        "unicorn in enchanted forest, magical particles, moonlight, fantasy illustration",
    ],
    "retro": [
        "1980s neon aesthetic, retrowave, palm trees silhouette, sunset grid, vaporwave",
        "vintage arcade cabinet, pixel art screen, dim lighting, nostalgic atmosphere",
        "1950s diner interior, chrome details, checkered floor, americana, retro photography",
        "cassette tape with tangled film, macro shot, 80s nostalgia, soft focus",
        "old film camera on wooden table, vintage lenses, warm lighting, retro vibes",
    ],
    "surreal": [
        "melting clocks on twisted trees, salvador dali style, dreamscape, surrealism",
        "floating islands connected by waterfalls, impossible physics, fantasy surrealism",
        "door in the middle of ocean, lonely chair, minimalist surrealism, mysterious",
        "giant teacup city, tiny houses inside, whimsical, alice in wonderland aesthetic",
        "staircase to nowhere in clouds, MC escher inspired, impossible architecture",
    ],
    "seasons": [
        "autumn forest path, fallen leaves carpet, warm colors, golden afternoon light",
        "spring cherry blossom avenue, pink petals falling, soft focus, romantic",
        "winter wonderland, frozen lake, snow-covered trees, blue hour, peaceful",
        "summer beach sunrise, pastel sky, calm ocean, minimalist seascape",
        "harvest scene, wheat fields, sunset, rural landscape, pastoral beauty",
    ],
    "portraits": [
        "wise elderly person portrait, weathered face, storytelling eyes, natural light",
        "child laughing with paint on face, joy, candid moment, colorful, vibrant",
        "warrior woman with war paint, fierce expression, dramatic lighting, powerful",
        "astronaut helmet reflection, earth in visor, space suit detail, cinematic",
        "musician lost in performance, stage lights, emotion, concert photography",
    ],
    "macro": [
        "water droplet on leaf, macro photography, morning dew, bokeh background",
        "butterfly wing scales, extreme closeup, iridescent colors, scientific macro",
        "snowflake crystal structure, macro detail, winter, blue tones, delicate",
        "soap bubble surface, rainbow reflection, macro, abstract patterns",
        "spider web with dew drops, backlit, golden hour, nature macro",
    ],
    "minimalist": [
        "single red balloon against blue sky, minimalist composition, simple beauty",
        "lone tree on hill, negative space, minimalist landscape, serene",
        "geometric shapes, flat colors, bauhaus style, modern minimalism",
        "zen garden sand patterns, stones, minimalist meditation, peaceful",
        "single feather floating, white background, minimalist still life, elegant",
    ],
    "food": [
        "gourmet dessert plating, chocolate art, restaurant photography, elegant",
        "farmers market produce, colorful vegetables, rustic wooden crate, fresh",
        "coffee art latte, foam design, cozy cafe, overhead shot, inviting",
        "sushi platter, artistic arrangement, japanese cuisine, clean presentation",
        "homemade bread loaf, steam rising, rustic kitchen, warm golden light",
    ],
    "weather": [
        "lightning storm over city, long exposure, dramatic weather, powerful nature",
        "fog rolling through valley, mountains in background, moody atmosphere",
        "rainbow after rain, countryside, puddles reflecting, hope and beauty",
        "tornado forming in distance, ominous sky, storm chaser perspective",
        "sun rays breaking through clouds, god rays, dramatic sky, inspiring",
    ],
    "cultural": [
        "japanese temple in autumn, red maple leaves, traditional architecture, peaceful",
        "moroccan marketplace, colorful spices, lanterns, bustling atmosphere",
        "indian holi festival, color powder explosion, celebration, vibrant energy",
        "african savanna sunset, acacia trees, wildlife silhouettes, majestic",
        "scandinavian winter cabin, northern lights, snow, cozy hygge aesthetic",
    ],
    "experimental": [
        "double exposure portrait, cityscape overlay, artistic photography, creative",
        "long exposure light painting, swirls of color, dark background, abstract",
        "tilt-shift miniature effect, city from above, toy-like, unique perspective",
        "infrared photography, pink foliage, surreal landscape, otherworldly",
        "prism light refraction, rainbow spectrum, glass art, abstract beauty",
    ],
}


def generate_random_images(num_images: int = 5, category: str | None = None):
    """Generate random images from diverse prompts.

    Args:
        num_images: Number of images to generate
        category: Specific category to use, or None for random selection
    """
    logger.info("random_generation_started", num_images=num_images, category=category)

    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = load_config(config_path)

    # Initialize generator
    generator = ImageGenerator(
        model_id=config.model.base_model,
        device=config.model.device,
    )

    logger.info("loading_model", model=config.model.base_model)
    generator.load_model()

    # Initialize gallery
    gallery_dir = Path(__file__).parent.parent / "gallery"
    gallery = GalleryManager(gallery_dir=gallery_dir)

    # Select prompts
    if category and category in RANDOM_PROMPTS:
        available_prompts = RANDOM_PROMPTS[category].copy()
        logger.info("using_category", category=category, prompts=len(available_prompts))
    else:
        # Mix from all categories
        available_prompts = []
        for cat_prompts in RANDOM_PROMPTS.values():
            available_prompts.extend(cat_prompts)
        logger.info("using_all_categories", total_prompts=len(available_prompts))

    # Shuffle to ensure randomness
    random.shuffle(available_prompts)

    # Generate images
    for i in range(num_images):
        if not available_prompts:
            logger.warning("no_more_prompts", generated=i)
            break

        prompt = available_prompts.pop()
        logger.info(
            "generating_image",
            index=i + 1,
            total=num_images,
            prompt=prompt[:50] + "...",
        )

        try:
            # Random parameters for variety
            steps = random.choice([20, 25, 30, 35])
            guidance = round(random.uniform(6.5, 8.5), 1)
            seed = random.randint(0, 999999)

            logger.info("generation_params", steps=steps, guidance=guidance, seed=seed)

            # Generate
            images = generator.generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, watermark, text, signature",
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed,
                num_images=1,
            )

            # Save to gallery
            for img in images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create filename from first few words of prompt
                prompt_slug = "_".join(prompt.split()[:3]).lower()
                prompt_slug = "".join(c for c in prompt_slug if c.isalnum() or c == "_")
                filename = f"random_{prompt_slug}_{timestamp}.png"

                filepath = gallery.gallery_dir / "2026" / "01" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)

                img.save(filepath)
                logger.info("image_saved", path=str(filepath), size=img.size)

                print(f"\n✅ Generated {i + 1}/{num_images}: {filename}")
                print(f"   📝 Prompt: {prompt[:70]}...")
                print(f"   ⚙️  Steps: {steps}, Guidance: {guidance}, Seed: {seed}")

        except Exception as e:
            logger.error("generation_failed", error=str(e), prompt=prompt)
            print(f"\n❌ Failed to generate image {i + 1}: {e}")
            continue

    logger.info("random_generation_complete", total_generated=num_images)
    print(f"\n🎉 Complete! Generated {num_images} random images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate random diverse images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available categories:
  {", ".join(RANDOM_PROMPTS.keys())}

Examples:
  # Generate 5 random images from any category
  python scripts/generate_random.py

  # Generate 10 cosmic-themed images
  python scripts/generate_random.py -n 10 -c cosmic

  # Generate 3 surreal images
  python scripts/generate_random.py -n 3 --category surreal
        """,
    )

    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        default=5,
        help="Number of images to generate (default: 5)",
    )

    parser.add_argument(
        "-c",
        "--category",
        type=str,
        choices=list(RANDOM_PROMPTS.keys()),
        help="Specific category to use (optional)",
    )

    args = parser.parse_args()

    print("🎨 AI Artist - Random Image Generator")
    print("=" * 50)
    if args.category:
        print(f"📂 Category: {args.category}")
    else:
        print("📂 Category: All (random mix)")
    print(f"🔢 Number of images: {args.num_images}")
    print("=" * 50)

    generate_random_images(num_images=args.num_images, category=args.category)
