#!/usr/bin/env python3
"""Generate an expanded collection with fresh creative prompts.

A follow-up to the ultimate collection with all-new prompt categories.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_artist.main import AIArtist
from src.ai_artist.utils.config import load_config
from src.ai_artist.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# Fresh creative prompt collection
EXPANDED_PROMPTS = {
    "cinematic_moments": [
        "hero standing alone on cliff edge, dramatic backlighting, epic movie scene",
        "rain-soaked noir detective in neon city, trench coat, mysterious atmosphere",
        "ancient library with floating books, magical dust particles, golden hour light",
        "lone samurai in bamboo forest, mist swirling, cinematic composition",
        "astronaut discovering alien ruins, otherworldly landscape, sense of wonder",
        "grand ballroom frozen in time, ghostly dancers, ethereal and haunting",
        "warrior facing massive dragon, scale contrast, fantasy epic battle",
        "explorer in bioluminescent cave system, glowing crystals, adventure scene",
    ],
    "emotional_portraits": [
        "elderly artist painting at sunset, hands covered in paint, life reflection",
        "child wonder discovering fireflies at dusk, magical moment, pure joy",
        "musician lost in performance, eyes closed, emotional intensity",
        "gardener with weathered hands holding fresh blooms, connection to nature",
        "dancer stretching at dawn, city skyline background, dedication and grace",
        "blacksmith at forge, sparks flying, proud craftsmanship",
        "lighthouse keeper on stormy night, lantern glow, solitary guardian",
        "chef plating masterpiece, intense concentration, culinary artistry",
    ],
    "nature_spectacles": [
        "massive waterfall in tropical rainforest, rainbow mist, raw power of nature",
        "ancient redwood forest with shafts of morning light, cathedral atmosphere",
        "volcanic eruption at night, red lava meeting ocean, primal forces",
        "rolling lavender fields in provence, purple waves, peaceful pastoral",
        "grand canyon at golden hour, layered rock formations, geological wonder",
        "monsoon clouds over desert, dramatic sky, storm approaching",
        "ice caves with blue crystalline walls, otherworldly frozen beauty",
        "coral reef teeming with colorful fish, underwater paradise, vibrant life",
    ],
    "urban_poetry": [
        "empty subway platform at 3am, single light flickering, urban solitude",
        "rooftop garden in megacity, green oasis, contrast of nature and concrete",
        "old bookstore on rainy night, warm interior glow, cozy refuge",
        "street food vendor in tokyo alley, steam rising, neon signs reflecting",
        "abandoned amusement park reclaimed by nature, melancholic beauty",
        "jazz club interior, saxophone player silhouetted, smoky ambiance",
        "street art mural being painted, urban canvas, creative expression",
        "morning market in marrakech, colorful spices, vibrant cultural tapestry",
    ],
    "dreamscapes": [
        "paper boat sailing on clouds, childlike wonder, soft pastel dream",
        "grandfather clock melting into sand, time flowing away, surreal",
        "library where books grow on trees, knowledge garden, whimsical concept",
        "stairway ascending into starfield, cosmic journey, metaphysical",
        "teacup containing entire ocean with sailing ships, impossible scale",
        "door carved into ancient tree leading to magical realm, portal fantasy",
        "musician playing piano made of water, fluid melody visualization",
        "painter whose brush strokes become real butterflies, art coming alive",
    ],
    "historical_reimagined": [
        "steampunk victorian london, brass airships, alternate history aesthetic",
        "samurai in cyberpunk tokyo, tradition meets future, neon and katana",
        "renaissance inventor workshop, da vinci style, mechanical marvels",
        "egyptian temple with holographic hieroglyphs, ancient meets advanced tech",
        "viking longship with solar sails, futuristic norse exploration",
        "medieval castle with bioluminescent gardens, fantasy medieval fusion",
        "roman colosseum hosting drone races, classical architecture repurposed",
        "art deco underwater city, retro futuristic atlantis, 1920s meets ocean",
    ],
    "seasonal_moods": [
        "first snow on japanese zen garden, peaceful transformation, winter serenity",
        "spring meadow explosion of wildflowers, bees busy, life returning",
        "autumn bonfire gathering, friends silhouettes, harvest celebration",
        "summer storm approaching wheat field, golden against dark clouds, drama",
        "winter cabin smoke curling from chimney, cozy refuge, snowy isolation",
        "spring cherry blossoms falling on traditional tea ceremony, transient beauty",
        "autumn fog rolling through vineyard at dawn, mysterious morning",
        "summer night beach with bioluminescent waves, natural magic, romance",
    ],
    "animal_majesty": [
        "white wolf in snowy forest, piercing blue eyes, wild nobility",
        "elephant herd at watering hole, golden dust, family bonds",
        "eagle soaring above mountain peaks, freedom personified, majestic flight",
        "tiger reflected in still jungle pool, perfect symmetry, powerful grace",
        "humpback whale breaching at sunset, ocean giant, awe-inspiring moment",
        "peacock displaying full plumage, iridescent blues and greens, natural art",
        "owl perched in moonlit tree, wise observer, nocturnal guardian",
        "dolphin pod jumping in synchronized arc, ocean joy, playful intelligence",
    ],
    "abstract_concepts": [
        "chaos and order collision, geometric shapes versus organic flow",
        "time visualization, clock faces dissolving into sand streams",
        "music made visible, sound waves forming colorful patterns",
        "thought processes, neural pathways glowing like city lights",
        "emotion spectrum, abstract color gradient with human silhouette",
        "memory fragments, photographs floating in misty void, nostalgia",
        "digital consciousness, binary code forming human face, AI awakening",
        "interconnectedness, golden threads linking all living things",
    ],
    "food_art": [
        "molecular gastronomy dish being plated, artistic precision, culinary innovation",
        "rustic italian pasta making, flour dusted hands, traditional craft",
        "japanese sushi master at work, knife precision, culinary meditation",
        "french patisserie window display, perfect pastries, mouthwatering artistry",
        "farm to table harvest spread, organic bounty, natural abundance",
        "chocolate fountain in artisan workshop, liquid gold, sweet craftsmanship",
        "spice market close-up, colorful pyramids of powder, aromatic visual",
        "sourdough bread being scored, artisan baker, simple perfection",
    ],
    "sci_fi_worlds": [
        "colony dome on mars, red planet settlement, human perseverance",
        "massive space station orbiting ringed planet, orbital city, future home",
        "terraforming in progress, green spreading across barren world, hope",
        "first contact moment, alien and human reaching out, historic meeting",
        "generation ship interior, forests and waterfalls, journey between stars",
        "quantum computer core, impossible geometry, technological singularity",
        "cyborg meditation in zen garden, humanity and tech harmonized",
        "time traveler library, doors to different eras, temporal archive",
    ],
    "mystical_elements": [
        "crystal cave with wizard studying ancient runes, magical scholarship",
        "fairy ring in moonlit forest, mushroom circle glowing, fae portal",
        "floating meditation, monk levitating surrounded by lotus flowers",
        "northern lights forming dragon shape, sky magic, aurora mythology",
        "enchanted forest pool with healing properties, ethereal mist rising",
        "ancient tree with face, forest guardian awakening, nature spirit",
        "witch brewing potion, colorful smoke swirling, mystical crafting",
        "celestial being descending on light beam, divine visitation, holy moment",
    ],
    "adventure_scenes": [
        "treasure hunter discovering golden city in jungle, lost civilization",
        "mountain climber reaching summit at sunrise, triumph and exhaustion",
        "deep sea diver encountering shipwreck, underwater exploration, mystery",
        "spelunker in vast underground cavern, headlamp illuminating vastness",
        "safari photographer eye to eye with lion, respect and danger",
        "paraglider soaring over coastline, freedom and exhilaration",
        "archaeologist uncovering dinosaur fossil, paleontology discovery",
        "storm chaser capturing tornado, dangerous pursuit, nature's fury",
    ],
    "cozy_scenes": [
        "reading nook with rain on window, warm tea, perfect afternoon",
        "fireplace crackling, cat sleeping, knitting supplies, domestic peace",
        "bakery early morning, fresh bread aroma, warm inviting light",
        "library study carrel, stacked books, focused student, academic sanctuary",
        "cottage kitchen baking cookies, grandmother teaching child, family warmth",
        "record player spinning vinyl, cozy apartment, music nostalgia",
        "greenhouse workshop, plants everywhere, potter at wheel, creative space",
        "mountain cabin interior, wood stove, snow outside, refuge from cold",
    ],
    "movement_frozen": [
        "ballet dancer mid-leap, fabric flowing, suspended grace",
        "water drop crown splash, macro photography, liquid sculpture",
        "bird taking flight, wings spread, feathers detailed, motion captured",
        "skateboarder mid-trick, urban backdrop, youth culture, frozen action",
        "rain droplets suspended in air, backlit, time stopped",
        "martial artist breaking board, power and discipline, peak moment",
        "horse galloping, mane flowing, muscles defined, athletic beauty",
        "paint being thrown, color explosion frozen, chaos organized",
    ],
    "light_studies": [
        "cathedral interior, stained glass colors on stone floor, sacred geometry",
        "desert sunrise, long shadows, warm orange light painting dunes",
        "bioluminescent plankton on beach at night, blue glow, natural wonder",
        "fiber optic cables glowing, technology art, modern light sculpture",
        "candles in darkness, multiple flames, warm intimate lighting",
        "lightning illuminating storm clouds from within, nature's electricity",
        "prism splitting sunlight into rainbow, physics made beautiful",
        "city lights reflected in rain puddles, urban mirror, night glow",
    ],
    "architectural_wonders": [
        "gothic cathedral reaching toward sky, flying buttresses, medieval majesty",
        "futuristic curved skyscraper, parametric design, architectural innovation",
        "ancient temple overgrown with jungle, nature reclaiming, time passage",
        "modernist house with infinity pool, minimalist luxury, architectural dream",
        "bridge spanning dramatic gorge, engineering marvel, human achievement",
        "opera house at night, dramatic lighting, cultural landmark",
        "traditional japanese house in snow, sliding doors, harmonious design",
        "art nouveau building facade, organic forms, architectural artistry",
    ],
    "texture_focus": [
        "tree bark close-up, intricate patterns, natural complexity",
        "weathered wood planks, grain and knots, aged beauty",
        "cracked desert mud patterns, geometric nature, drought artistry",
        "fabric macro, woven threads, textile detail",
        "rusted metal with peeling paint, industrial decay, abstract texture",
        "feather detail, barbs and barbules, natural engineering",
        "stone wall weathered by centuries, layers of history, tactile time",
        "ice crystals forming on glass, fractal patterns, frozen art",
    ],
    "cultural_celebrations": [
        "holi festival, colored powder explosion, joyful chaos, vibrant tradition",
        "dia de los muertos altar, marigolds and photos, honoring ancestors",
        "chinese new year dragon dance, red and gold, cultural pageantry",
        "carnival in rio, elaborate costumes, celebration and music",
        "diwali oil lamps at dusk, rows of lights, festival of light",
        "scottish highland games, kilts and cabers, tradition and strength",
        "venetian masquerade ball, ornate masks, mysterious elegance",
        "midsummer bonfire ritual, flames leaping, ancient celebration",
    ],
    "minimalist_beauty": [
        "single bonsai on zen table, negative space, mindful simplicity",
        "lone figure in vast white desert, scale and isolation",
        "single red leaf on snow, color contrast, seasonal transition",
        "minimal concrete architecture, clean lines, modernist purity",
        "single boat on misty lake, peaceful solitude, minimalist serenity",
        "black ink calligraphy on white, elegant simplicity, artistic restraint",
        "desert road vanishing to horizon, endless perspective, minimal landscape",
        "single flower in empty room, focal point, beauty in simplicity",
    ],
}


def count_all_prompts() -> int:
    """Count total available prompts."""
    return sum(len(prompts) for prompts in EXPANDED_PROMPTS.values())


def collect_prompts(categories: list = None, randomize: bool = False):
    """Collect prompts from specified categories."""
    available_prompts = []

    # Filter categories if specified
    prompt_dict = EXPANDED_PROMPTS
    if categories:
        prompt_dict = {k: v for k, v in EXPANDED_PROMPTS.items() if k in categories}

    # Collect all prompts with their categories
    for category, prompts in prompt_dict.items():
        for prompt in prompts:
            available_prompts.append((category, prompt))

    # Randomize if requested
    if randomize:
        random.shuffle(available_prompts)

    return available_prompts


def get_generation_params(vary_parameters: bool) -> dict:
    """Get generation parameters with optional variation."""
    if vary_parameters:
        return {
            "steps": random.choice([20, 25, 30]),
            "guidance": round(random.uniform(7.0, 8.0), 1),
            "seed": random.randint(0, 999999),
        }
    else:
        return {
            "steps": 25,
            "guidance": 7.5,
            "seed": None,
        }


def print_header(total: int, artist: AIArtist, randomize: bool, vary_parameters: bool):
    """Print generation header."""
    print(f"\n{'=' * 70}")
    print("🎨 AI ARTIST - EXPANDED COLLECTION GENERATOR")
    print(f"{'=' * 70}")
    print("📊 Statistics:")
    print(f"   • Total categories: {len(EXPANDED_PROMPTS)}")
    print(f"   • Available prompts: {count_all_prompts()}")
    print(f"   • Generating: {total} artworks")
    print(f"   • Device: {artist.generator.device}")
    print(f"   • Model: {artist.generator.model_id}")
    print(f"   • Randomized: {'Yes' if randomize else 'No'}")
    print(f"   • Parameter variation: {'Yes' if vary_parameters else 'No'}")
    print(f"{'=' * 70}\n")


def print_progress(
    i: int, total: int, category: str, prompt: str, params: dict, vary: bool
):
    """Print progress for current generation."""
    print(f"[{i}/{total}] Category: {category}")
    print(f"   Prompt: {prompt[:65]}...")
    if vary:
        print(
            f"   Params: steps={params['steps']}, guidance={params['guidance']}, seed={params['seed']}"
        )


def print_summary(
    success_count: int, total: int, failed_prompts: list, category_counts: dict
):
    """Print final summary."""
    print(f"\n{'=' * 70}")
    print("✨ GENERATION COMPLETE!")
    print(f"{'=' * 70}")
    print("📊 Results:")
    print(
        f"   ✅ Successful: {success_count}/{total} ({success_count / total * 100:.1f}%)"
    )
    print(f"   ❌ Failed: {len(failed_prompts)}")
    print("\n📁 Category breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {cat}: {count} images")

    if failed_prompts:
        print("\n❌ Failed prompts:")
        for cat, prompt in failed_prompts:
            print(f"   • [{cat}] {prompt[:50]}...")

    print("\n💾 All images saved to gallery/2026/")
    print(f"{'=' * 70}\n")


def generate_expanded_collection(
    num_images: int = None,
    categories: list = None,
    randomize: bool = True,
    vary_parameters: bool = True,
):
    """Generate expanded collection with fresh prompts.

    Args:
        num_images: Number of images to generate (None = all prompts)
        categories: Specific categories to use (None = all categories)
        randomize: Whether to shuffle prompts randomly
        vary_parameters: Whether to vary generation parameters for variety
    """
    logger.info(
        "expanded_collection_started",
        num_images=num_images,
        categories=categories,
        randomize=randomize,
    )

    # Load configuration
    config_path = Path("config/config.yaml")
    config = load_config(config_path)

    # Initialize AI Artist
    artist = AIArtist(config)

    # Detect best available device
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        logger.info("using_cuda_device")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        logger.info("using_mps_device")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.info("using_cpu_device")

    # Configure generator for optimal performance
    artist.generator.device = device
    artist.generator.dtype = dtype
    artist.generator.model_id = "runwayml/stable-diffusion-v1-5"

    # Load model
    artist.generator.load_model()

    # Collect and prepare prompts
    available_prompts = collect_prompts(categories, randomize)

    # Limit to requested number
    if num_images:
        available_prompts = available_prompts[:num_images]

    total = len(available_prompts)

    # Display header
    print_header(total, artist, randomize, vary_parameters)

    # Generation statistics
    success_count = 0
    failed_prompts = []
    category_counts = {}

    # Generate images
    for i, (category, prompt) in enumerate(available_prompts, 1):
        try:
            # Track category stats
            category_counts[category] = category_counts.get(category, 0) + 1

            # Get generation parameters
            params = get_generation_params(vary_parameters)

            # Display progress
            print_progress(i, total, category, prompt, params, vary_parameters)

            # Generate image with proper resolution for SD 1.5
            images = artist.generator.generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, watermark, text, signature, bad anatomy",
                width=512,
                height=512,
                num_images=1,
                num_inference_steps=params["steps"],
                guidance_scale=params["guidance"],
                seed=params["seed"],
            )

            # Save to gallery
            if images:
                metadata = {
                    "prompt": prompt,
                    "category": category,
                    "model": artist.generator.model_id,
                    "steps": params["steps"],
                    "guidance_scale": params["guidance"],
                    "seed": params["seed"] if params["seed"] else "random",
                    "collection": "expanded",
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
                failed_prompts.append((category, prompt))
                print("   ❌ Generation failed\n")

        except KeyboardInterrupt:
            logger.info("generation_interrupted_by_user")
            print("\n\n⚠️  Generation interrupted by user")
            break
        except Exception as e:
            logger.error(
                "generation_failed", category=category, prompt=prompt[:50], error=str(e)
            )
            failed_prompts.append((category, prompt))
            print(f"   ❌ Error: {e}\n")
            continue

    # Print final summary
    print_summary(success_count, total, failed_prompts, category_counts)

    # Cleanup
    artist.generator.unload()
    logger.info(
        "expanded_collection_complete",
        success=success_count,
        failed=len(failed_prompts),
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate expanded art collection with fresh creative prompts"
    )
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        help="Number of images to generate (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all prompts (same as omitting -n)",
    )
    parser.add_argument(
        "-c",
        "--categories",
        nargs="+",
        help="Specific categories to generate from",
    )
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="Don't randomize prompt order",
    )
    parser.add_argument(
        "--no-vary",
        action="store_true",
        help="Don't vary generation parameters",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories and exit",
    )

    args = parser.parse_args()

    if args.list_categories:
        print("\n📋 Available Categories:\n")
        for i, (category, prompts) in enumerate(EXPANDED_PROMPTS.items(), 1):
            print(f"{i:2}. {category:25} ({len(prompts)} prompts)")
        print(
            f"\nTotal: {len(EXPANDED_PROMPTS)} categories, {count_all_prompts()} prompts\n"
        )
        return

    generate_expanded_collection(
        num_images=args.num_images,
        categories=args.categories,
        randomize=not args.no_randomize,
        vary_parameters=not args.no_vary,
    )


if __name__ == "__main__":
    main()
