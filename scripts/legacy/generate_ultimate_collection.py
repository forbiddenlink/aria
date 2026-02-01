#!/usr/bin/env python3
"""Generate an ultimate diverse collection with 200+ creative prompts.

This script combines and expands upon all previous generation scripts,
offering the most comprehensive collection of prompt categories and themes.
"""

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_artist.main import AIArtist
from src.ai_artist.utils.config import load_config
from src.ai_artist.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# Comprehensive prompt collection organized by theme
ULTIMATE_PROMPTS = {
    "cosmic_wonders": [
        "spiral galaxy with vibrant nebula, deep space photography, cosmic dust clouds, 4k astronomy",
        "black hole event horizon, warped spacetime, glowing accretion disk, scientific visualization",
        "twin binary stars orbiting, stellar corona, solar flares, space telescope image",
        "supernova explosion, shockwave expanding, cosmic debris, colorful gas clouds",
        "alien exoplanet with three moons, gas giant rings, purple atmosphere, sci-fi realism",
        "asteroid belt, rocky debris field, distant sun, epic space vista, cinematic",
        "pulsar neutron star, magnetic field lines, high energy beams, scientific art",
        "planetary nebula, dying star, colorful expanding shells, deep field image",
    ],
    "mythological_beings": [
        "phoenix rising from golden flames, glowing embers, mythical firebird, epic wings spread",
        "ancient dragon perched on castle ruins, detailed scales, moonlit night, fantasy epic",
        "mermaid queen in bioluminescent ocean depths, coral throne, ethereal underwater beauty",
        "majestic unicorn in enchanted forest, magical sparkles, moonbeams filtering through trees",
        "thunderbird summoning storm, lightning crackling, native american mythology, powerful",
        "kraken emerging from stormy seas, massive tentacles, sailing ship, dramatic waves",
        "griffin guardian of ancient temple, lion and eagle hybrid, golden hour light",
        "basilisk serpent with crown, poisonous gaze, dark cavern, mythical monster art",
    ],
    "retro_aesthetics": [
        "1980s neon cityscape, retrowave sunset, palm trees silhouette, vaporwave pink and purple",
        "vintage arcade cabinet glowing, pixel art game, dim room, nostalgic 80s atmosphere",
        "1950s american diner, chrome fixtures, checkered floor, jukebox playing, classic cars outside",
        "retro cassette tape, tangled magnetic film, macro closeup, 80s music nostalgia",
        "old polaroid camera, instant photos scattered, vintage aesthetic, warm analog feel",
        "neon signs reflecting in rain puddles, blade runner style, cyberpunk retro, night city",
        "vintage transistor radio, wood paneling, glowing dial, 1970s living room",
        "classic jukebox with vinyl records, colorful lights, 1960s soda shop interior",
    ],
    "surreal_dreams": [
        "melting clocks draped over twisted trees, salvador dali inspired, dreamscape reality",
        "floating islands connected by impossible waterfalls, clouds below, surreal physics",
        "single door standing in vast desert, no walls, mysterious portal, minimalist surrealism",
        "giant teacup containing miniature city, tiny people, whimsical alice wonderland style",
        "infinite staircase spiraling in clouds, MC escher impossible geometry, mind-bending",
        "man in bowler hat with apple obscuring face, magritte surrealism, mysterious portrait",
        "bedroom ceiling covered with ocean waves, underwater room, reality inversion",
        "tree growing upside down from sky, roots in clouds, leaves underground, paradox",
    ],
    "four_seasons": [
        "autumn forest path, carpet of red and gold leaves, misty morning, peaceful serenity",
        "spring cherry blossom tunnel, pink petals falling like snow, soft romantic light",
        "winter wonderland, frozen waterfall, icicles, snow-covered evergreens, blue hour magic",
        "summer sunflower field at golden hour, warm sunset glow, pastoral landscape",
        "fall harvest scene, pumpkin patch, hay bales, orange autumn sky, rustic farm",
        "spring rain shower, fresh flowers blooming, raindrops on petals, renewal and growth",
        "winter northern lights, snow-covered log cabin, aurora borealis dancing, starry night",
        "summer beach at sunrise, pastel sky, calm turquoise water, peaceful morning meditation",
    ],
    "portrait_characters": [
        "wise elderly wizard reading ancient spellbook, candlelight, long white beard, magical study",
        "cyberpunk hacker with neon tattoos, city lights reflected in eyes, rain, futuristic",
        "steampunk inventor with brass goggles, workshop full of gears, victorian aesthetic",
        "tribal warrior with ceremonial paint, fierce expression, cultural photography, powerful",
        "ballerina mid-leap in flowing white dress, stage lights, graceful frozen motion",
        "astronaut portrait, earth reflection in helmet visor, stars, heroic space explorer",
        "geisha in traditional kimono, kyoto streets, cherry blossoms, cultural elegance",
        "viking shield maiden with braided hair, war paint, fierce nordic warrior, dramatic",
    ],
    "macro_mysteries": [
        "dewdrop on spider web at sunrise, backlit golden light, macro detail, nature close-up",
        "butterfly wing scales extreme macro, iridescent colors, scientific photography",
        "snowflake crystal structure on dark wool, geometric patterns, winter macro beauty",
        "soap bubble surface, rainbow swirls, macro abstract, delicate transparent sphere",
        "bee covered in yellow pollen, flower stamens, macro wildlife, pollination moment",
        "watch mechanism gears, intricate clockwork, steampunk macro, mechanical precision",
        "peacock feather eye, microscopic detail, vibrant blues and greens, natural art",
        "salt crystals forming patterns, macro closeup, geometric mineral structures, abstract",
    ],
    "minimalist_zen": [
        "single red balloon floating against clear blue sky, minimalist composition, simple joy",
        "solitary tree on rolling green hill, negative space, fog, zen landscape meditation",
        "geometric bauhaus shapes, primary colors, flat design, modern minimalist art",
        "zen garden sand ripples, three stones, minimalist japanese aesthetic, peaceful",
        "single feather falling through white space, minimal still life, delicate grace",
        "lone boat on calm lake, mirror reflection, pastel dawn, serene minimalism",
        "abstract color field, rothko inspired, soft edge rectangles, contemplative art",
        "single leaf floating on water, concentric ripples, zen moment, natural minimalism",
    ],
    "culinary_art": [
        "gourmet chocolate dessert, gold leaf decoration, artistic plating, fine dining photography",
        "colorful macarons arranged on marble, pastel rainbow, french patisserie, overhead shot",
        "steaming ramen bowl, chopsticks, soft-boiled egg, japanese cuisine, cozy atmosphere",
        "fresh berries in antique bowl, rustic wooden table, natural window light, farm fresh",
        "artisan sourdough bread, crust detail, steam rising, bakery atmosphere, warm light",
        "sushi chef crafting nigiri, precision cutting, fresh fish, culinary art performance",
        "italian pasta making, flour dust, hands kneading dough, rustic kitchen, tradition",
        "coffee latte art, heart foam design, ceramic cup, cafe morning light, inviting",
    ],
    "weather_phenomena": [
        "lightning bolt striking ocean during storm, long exposure, dramatic power of nature",
        "morning fog rolling through mountain valley, layers of hills, moody atmosphere",
        "double rainbow over green countryside, rain clearing, hope and natural beauty",
        "tornado funnel forming in wheat field, ominous dark clouds, storm chaser perspective",
        "sun rays breaking through storm clouds, crepuscular rays, divine light, inspiring",
        "mammatus clouds, unusual formations, severe weather, dramatic sky, natural wonder",
        "ice storm coating trees, crystalline branches, winter wonderland, frozen beauty",
        "dust devil in arizona desert, red rock landscape, swirling sand, wild weather",
    ],
    "architectural_marvels": [
        "futuristic glass skyscraper, reflective facade, blue sky, modern architectural photography",
        "ancient gothic cathedral interior, stained glass windows, divine light rays, sacred space",
        "japanese temple in autumn, red torii gate, maple leaves, traditional architecture",
        "modern minimalist house, concrete and glass, infinity pool, architectural digest style",
        "moroccan palace courtyard, ornate zellige tiles, fountain, arches, warm sunlight",
        "brutalist concrete building, geometric shapes, soviet architecture, dramatic angles",
        "art deco building facade, gold accents, 1920s elegance, miami beach style",
        "bamboo forest temple, zen architecture, natural materials, harmony with nature",
    ],
    "urban_exploration": [
        "tokyo street at night, neon signs reflecting in rain, cyberpunk atmosphere, cinematic",
        "abandoned factory interior, rust and decay, urban exploration, dramatic shafts of light",
        "cobblestone european alley, flower boxes, cafe tables, golden afternoon, romantic",
        "new york city rooftop view, sunset skyline, urban photography, metropolis energy",
        "street art mural, colorful graffiti, urban culture, large scale wall painting",
        "night market in bangkok, food stalls, lanterns, bustling atmosphere, street photography",
        "london fog, big ben clock tower, red telephone booth, moody urban atmosphere",
        "dubai skyline at dusk, illuminated skyscrapers, desert modern, luxury cityscape",
    ],
    "wildlife_moments": [
        "peacock displaying full tail feathers, vibrant iridescent colors, botanical garden",
        "arctic fox in snow, white fur, blue hour lighting, nature documentary style",
        "hummingbird frozen mid-flight, feeding on flower, macro, tropical colors",
        "elephant family at watering hole, golden sunset, african savanna, dramatic silhouette",
        "owl perched on branch, intense eyes, moonlit forest, wildlife portrait",
        "sea turtle swimming in coral reef, underwater, marine life, ocean conservation",
        "red panda in bamboo forest, cute expression, endangered species, natural habitat",
        "cheetah running at full speed, motion blur, hunting, wild africa, powerful grace",
    ],
    "fantasy_landscapes": [
        "crystal cave with glowing minerals, underground lake, magical geodes, fantasy exploration",
        "floating castle in clouds, waterfalls cascading down, epic fantasy architecture",
        "bioluminescent forest at night, glowing mushrooms, magical fairy tale atmosphere",
        "volcanic landscape, lava flows, ash clouds, primordial earth, dramatic geology",
        "ice palace in frozen tundra, aurora overhead, fantasy kingdom, winter magic",
        "giant sequoia forest, ancient trees, mystical fog, cathedral of nature",
        "desert oasis, palm trees, clear spring water, sand dunes, serene refuge",
        "Scottish highlands, misty mountains, castle ruins, heather fields, celtic atmosphere",
    ],
    "sci_fi_futures": [
        "space station interior, holographic displays, futuristic design, sci-fi concept art",
        "cyberpunk street market, neon signs, androids, rain reflections, blade runner aesthetic",
        "mars colony dome, red planet surface, terraforming, future human settlement",
        "steampunk airship, brass machinery, clouds, victorian era fusion, fantasy technology",
        "alien technology, glowing symbols, advanced civilization, mysterious artifacts",
        "post-apocalyptic cityscape, overgrown buildings, nature reclaiming, dramatic sky",
        "underwater city, geodesic domes, submarines, futuristic ocean colonization",
        "robot factory, assembly line, industrial sci-fi, mechanical precision, future industry",
    ],
    "abstract_expressions": [
        "fluid art, swirling marble patterns, gold and turquoise, abstract expressionism",
        "geometric mandala, intricate symmetrical patterns, vibrant gradients, spiritual art",
        "watercolor wash, soft color blending, pastel abstract, dreamy impressionism",
        "fractal mathematics visualization, infinite patterns, psychedelic colors, digital art",
        "jackson pollock style, action painting, splatter technique, abstract energy",
        "color field painting, rothko inspired, contemplative rectangles, emotional depth",
        "kinetic motion blur, light trails, abstract movement, dynamic energy",
        "molecular structure, atoms and bonds, scientific abstract, microscopic beauty",
    ],
    "still_life_classics": [
        "vintage typewriter, old books, coffee cup, wooden desk, writer's atmosphere",
        "fresh fruit bowl, dramatic chiaroscuro lighting, dutch masters style, oil painting aesthetic",
        "origami crane collection, pastel paper, soft shadows, japanese minimalist art",
        "antique pocket watch, gears visible, macro photography, steampunk details",
        "wilting roses in crystal vase, symbolism, memento mori, classical still life",
        "vinyl record player, spinning disc, warm lamp light, music nostalgia, analog",
        "seashell collection on driftwood, beach findings, natural history, coastal aesthetic",
        "vintage perfume bottles, art deco glass, vanity table, feminine elegance, soft focus",
    ],
    "underwater_worlds": [
        "coral reef ecosystem, tropical fish, sunlight filtering through water, marine biodiversity",
        "shipwreck covered in coral, underwater exploration, mysterious depths, diving photography",
        "jellyfish floating in dark water, bioluminescent glow, ethereal sea creatures",
        "dolphins playing in crystal clear ocean, underwater photography, marine mammals",
        "kelp forest, sea otters, underwater jungle, california coast, marine sanctuary",
        "manta ray gliding overhead, underwater perspective, graceful sea creature, ocean giant",
        "sea anemone closeup, clownfish, symbiotic relationship, macro marine life",
        "underwater cave, light shafts penetrating, cenote diving, mysterious blue depths",
    ],
    "cultural_celebrations": [
        "indian holi festival, explosion of color powder, celebration, joyful energy",
        "day of the dead altar, marigolds, sugar skulls, candles, mexican tradition",
        "chinese new year, red lanterns, dragon dance, fireworks, cultural celebration",
        "brazilian carnival, colorful costumes, feathers, samba dancers, festive energy",
        "japanese tea ceremony, matcha preparation, minimalist ritual, cultural meditation",
        "scottish highland games, kilts, bagpipes, athletic competition, celtic tradition",
        "venetian carnival mask, ornate decoration, mysterious elegance, italian tradition",
        "diwali lights, oil lamps, rangoli patterns, indian festival of lights, celebration",
    ],
    "time_of_day": [
        "golden hour meadow, warm sunlight, wildflowers, magic hour photography, peaceful",
        "blue hour cityscape, twilight, lights beginning to glow, transitional beauty",
        "midday harsh shadows, architectural photography, strong contrast, summer heat",
        "dawn breaking over ocean, first light, pastel sky, new beginning, hope",
        "midnight moon, stars visible, long exposure, night sky photography, cosmic",
        "afternoon tea time, soft window light, cozy interior, relaxing moment",
        "dusk silhouettes, sunset colors, dramatic sky, day ending, contemplative",
        "predawn darkness, last stars fading, anticipation of sunrise, quiet moment",
    ],
    "textural_studies": [
        "tree bark macro, detailed texture, natural patterns, organic surface, close-up",
        "rusted metal surface, oxidation patterns, industrial decay, abstract texture",
        "cracked earth, drought patterns, natural geometry, environmental texture",
        "fabric folds, silk material, light and shadow, textile study, elegant draping",
        "weathered wood grain, aged surface, natural wear, rustic texture, time's passage",
        "ice crystals forming patterns, frost on window, winter texture, natural art",
        "sand ripples, wind patterns, desert texture, natural waves, minimalist",
        "brick wall detail, mortar lines, architectural texture, urban surface, geometric",
    ],
    "emotional_atmospheres": [
        "lonely bench in fog, empty park, solitude, melancholy mood, contemplative",
        "joyful child running through sprinkler, summer fun, happiness, candid moment",
        "tense chess game, focused players, competitive atmosphere, intellectual battle",
        "serene meditation, peaceful pose, calm environment, inner peace, mindfulness",
        "anxious storm approaching, dark clouds, ominous atmosphere, tension, dramatic",
        "romantic sunset picnic, couple silhouette, love, intimate moment, warm colors",
        "nostalgic childhood bedroom, vintage toys, memories, sentimental, soft focus",
        "triumphant mountain summit, arms raised, achievement, inspiring, victory pose",
    ],
    "light_and_shadow": [
        "venetian blinds casting striped shadows, noir aesthetic, dramatic contrast, mystery",
        "spotlight on empty stage, dramatic lighting, theater atmosphere, anticipation",
        "stained glass window light patterns on floor, colorful shadows, sacred geometry",
        "forest sunbeams, light rays through trees, atmospheric fog, divine light",
        "silhouette against sunset, backlit figure, dramatic outline, golden hour",
        "chiaroscuro portrait, rembrandt lighting, dramatic shadow, classical technique",
        "long shadow at sunset, extended form, golden hour, artistic photography",
        "paper cutout shadow art, layered shadows, creative lighting, artistic installation",
    ],
    "magical_realism": [
        "books flying around library, magical literacy, whimsical, fantasy atmosphere",
        "garden where flowers bloom instantly, time-lapse effect, magical nature, wonder",
        "mirror reflecting different season, magical portal, reality shift, surreal",
        "stars falling from sky into jar, catching stardust, magical collection, dreamy",
        "painting coming to life, canvas reality blend, magical art, metamorphosis",
        "clock running backwards, time reversal, temporal magic, surreal concept",
        "footprints glowing in darkness, magical trail, bioluminescent path, fantasy night",
        "umbrella holding up rain cloud, whimsical weather, playful magic, cute surrealism",
    ],
}


def count_all_prompts() -> int:
    """Count total number of prompts across all categories."""
    return sum(len(prompts) for prompts in ULTIMATE_PROMPTS.values())


def _collect_prompts(
    categories: list[str] | None, randomize: bool
) -> list[tuple[str, str]]:
    """Collect and prepare prompts based on category selection.

    Args:
        categories: Specific categories to use (None = all categories)
        randomize: Whether to shuffle prompts randomly

    Returns:
        List of (category, prompt) tuples
    """
    if categories:
        available_prompts = []
        for cat in categories:
            if cat in ULTIMATE_PROMPTS:
                available_prompts.extend([(cat, p) for p in ULTIMATE_PROMPTS[cat]])
        logger.info(
            "using_categories", categories=categories, total=len(available_prompts)
        )
    else:
        available_prompts = []
        for cat, prompts in ULTIMATE_PROMPTS.items():
            available_prompts.extend([(cat, p) for p in prompts])
        logger.info("using_all_categories", total=len(available_prompts))

    if randomize:
        random.shuffle(available_prompts)

    return available_prompts


def _get_generation_params(vary_parameters: bool) -> dict:
    """Get generation parameters with optional variation.

    Args:
        vary_parameters: Whether to vary generation parameters

    Returns:
        Dictionary with steps, guidance_scale, and seed
    """
    if vary_parameters:
        return {
            "steps": random.choice([20, 25, 30]),  # Removed 15 - too low for quality
            "guidance": round(
                random.uniform(7.0, 8.0), 1
            ),  # Narrower range for consistency
            "seed": random.randint(0, 999999),
        }
    else:
        return {
            "steps": 25,  # Increased from 20 for better quality
            "guidance": 7.5,
            "seed": None,
        }


def _print_collection_header(
    total: int, artist: AIArtist, randomize: bool, vary_parameters: bool
):
    """Print the collection generation header."""
    print(f"\n{'=' * 70}")
    print("🎨 AI ARTIST - ULTIMATE COLLECTION GENERATOR")
    print(f"{'=' * 70}")
    print("📊 Statistics:")
    print(f"   • Total categories: {len(ULTIMATE_PROMPTS)}")
    print(f"   • Available prompts: {count_all_prompts()}")
    print(f"   • Generating: {total} artworks")
    print(f"   • Device: {artist.generator.device}")
    print(f"   • Model: {artist.generator.model_id}")
    print(f"   • Randomized: {'Yes' if randomize else 'No'}")
    print(f"   • Parameter variation: {'Yes' if vary_parameters else 'No'}")
    print(f"{'=' * 70}\n")


def _print_progress(
    i: int, total: int, category: str, prompt: str, params: dict, vary: bool
):
    """Print progress for current generation."""
    print(f"[{i}/{total}] Category: {category}")
    print(f"   Prompt: {prompt[:65]}...")
    if vary:
        print(
            f"   Params: steps={params['steps']}, guidance={params['guidance']}, seed={params['seed']}"
        )


def _print_final_summary(
    success_count: int, total: int, failed_prompts: list, category_counts: dict
):
    """Print final generation summary."""
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
        print("\n⚠️  Failed generations:")
        for idx, cat, prompt in failed_prompts[:10]:  # Show first 10
            print(f"   [{idx}] {cat}: {prompt[:50]}...")

    print(f"{'=' * 70}\n")


async def generate_ultimate_collection(
    num_images: int | None = None,
    categories: list[str] | None = None,
    randomize: bool = True,
    vary_parameters: bool = True,
):
    """Generate the ultimate diverse artwork collection.

    Args:
        num_images: Number of images to generate (None = all prompts)
        categories: Specific categories to use (None = all categories)
        randomize: Whether to shuffle prompts randomly
        vary_parameters: Whether to vary generation parameters for variety
    """
    logger.info(
        "ultimate_collection_started",
        num_images=num_images,
        categories=categories,
        randomize=randomize,
    )

    # Load configuration
    config_path = Path("config/config.yaml")
    config = load_config(config_path)

    # Initialize AI Artist
    artist = AIArtist(config)

    # Detect best available device for optimal performance
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
    available_prompts = _collect_prompts(categories, randomize)

    # Limit to requested number
    if num_images:
        available_prompts = available_prompts[:num_images]

    total = len(available_prompts)

    # Display header
    _print_collection_header(total, artist, randomize, vary_parameters)

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
            params = _get_generation_params(vary_parameters)

            # Display progress
            _print_progress(i, total, category, prompt, params, vary_parameters)

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
                    "collection": "ultimate",
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
                failed_prompts.append((i, category, prompt))

        except Exception as e:
            logger.error("generation_failed", error=str(e), prompt=prompt)
            print(f"   ❌ Error: {e}\n")
            failed_prompts.append((i, category, prompt))
            continue

    # Final summary
    _print_final_summary(success_count, total, failed_prompts, category_counts)

    if len(failed_prompts) > 10:
        print(f"   ... and {len(failed_prompts) - 10} more")

    print("\n🌐 View your gallery:")
    print("   • Web interface: http://localhost:8000")
    print(f"   • File location: {artist.gallery.gallery_dir}")
    print(f"{'=' * 70}\n")

    logger.info(
        "ultimate_collection_complete",
        success=success_count,
        failed=len(failed_prompts),
        categories=len(category_counts),
    )


if __name__ == "__main__":
    import argparse

    # Build category choices
    all_categories = list(ULTIMATE_PROMPTS.keys())

    parser = argparse.ArgumentParser(
        description="Generate the ultimate diverse art collection with 200+ prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
🎨 Available Categories ({len(all_categories)}):
  {", ".join(all_categories)}

📊 Total Available Prompts: {count_all_prompts()}

💡 Examples:
  # Generate all prompts from all categories
  python scripts/generate_ultimate_collection.py --all

  # Generate 50 random images
  python scripts/generate_ultimate_collection.py -n 50

  # Generate 20 images from specific categories
  python scripts/generate_ultimate_collection.py -n 20 -c cosmic_wonders mythological_beings

  # Generate without randomization (in order)
  python scripts/generate_ultimate_collection.py -n 30 --no-random

  # Generate with fixed parameters
  python scripts/generate_ultimate_collection.py -n 10 --no-variation
        """,
    )

    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        help="Number of images to generate (omit for all prompts)",
    )

    parser.add_argument(
        "-c",
        "--categories",
        type=str,
        nargs="+",
        choices=all_categories,
        help="Specific categories to use (omit for all)",
    )

    parser.add_argument(
        "--all", action="store_true", help="Generate all available prompts"
    )

    parser.add_argument(
        "--no-random", action="store_true", help="Don't randomize prompt order"
    )

    parser.add_argument(
        "--no-variation",
        action="store_true",
        help="Use fixed parameters instead of varying them",
    )

    parser.add_argument(
        "--list-categories", action="store_true", help="List all categories and exit"
    )

    args = parser.parse_args()

    # Handle list categories
    if args.list_categories:
        print(f"\n🎨 Available Categories ({len(all_categories)}):\n")
        for cat in sorted(all_categories):
            count = len(ULTIMATE_PROMPTS[cat])
            print(f"  • {cat:30} ({count} prompts)")
        print(f"\n📊 Total: {count_all_prompts()} prompts\n")
        sys.exit(0)

    # Determine number of images
    num_images = None
    if not args.all:
        num_images = args.num_images or 10  # Default to 10 if not specified

    # Run generation
    asyncio.run(
        generate_ultimate_collection(
            num_images=num_images,
            categories=args.categories,
            randomize=not args.no_random,
            vary_parameters=not args.no_variation,
        )
    )
