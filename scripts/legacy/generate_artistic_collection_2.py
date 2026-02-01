#!/usr/bin/env python3
"""Generate a second artistic collection with entirely new creative styles.

Fourth collection featuring modern art movements, experimental techniques, and creative visions.
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

# Second artistic prompt collection - completely new styles
ARTISTIC_PROMPTS_2 = {
    "japanese_aesthetics": [
        "ukiyo-e wave, hokusai great wave, woodblock print, prussian blue",
        "sakura petals falling, japanese painting style, gold leaf background",
        "geisha portrait, nihonga technique, mineral pigments, elegant composition",
        "zen rock garden from above, raked gravel patterns, minimalist beauty",
        "koi fish in pond, rimpa school style, decorative gold screens",
        "mount fuji with red sun, hiroshige landscape, atmospheric perspective",
        "kabuki actor portrait, bold makeup, dynamic pose, theatrical drama",
        "autumn leaves on stream, yamatoe painting, seasonal poetry visual",
    ],
    "contemporary_abstract": [
        "fluid acrylic pour, cellular patterns, organic color interactions",
        "resin art with gold leaf, glossy depths, luxury abstract",
        "spray paint gradients, urban contemporary, smooth color transitions",
        "marbled ink on water, suminagashi technique, controlled chaos",
        "encaustic wax layers, textured abstract, heat manipulation art",
        "alcohol ink blooms, organic spreading, vibrant color interactions",
        "pour painting with metallics, flowing liquid metal aesthetic",
        "abstract landscape suggestion, turner modern interpretation, atmospheric",
    ],
    "renaissance_masters": [
        "madonna and child, botticelli grace, tempera on panel, classical beauty",
        "creation hands almost touching, michelangelo fresco, divine moment",
        "last supper composition, leonardo perspective, dramatic narrative",
        "venus birth from sea, mythological scene, idealized form, classical",
        "self portrait with penetrating gaze, rembrandt technique, soul revealed",
        "classical architecture ruins, romantic landscape, claude lorrain light",
        "mythological transformation scene, ovid metamorphoses, baroque drama",
        "sfumato portrait, leonardo technique, mysterious smile, soft edges",
    ],
    "photo_realism": [
        "water droplets on glass, hyper-realistic detail, crystal clarity",
        "chrome reflection sphere, perfect mirror surface, photorealistic metal",
        "eye extreme close-up, iris detail, hyper-real texture and depth",
        "vintage car paint reflection, photorealistic chrome and lacquer",
        "still life with glass and fruit, dutch master realism, contemporary",
        "portrait with every pore visible, hyper-detail, photographic quality",
        "crumpled paper texture, shadow and highlight, trompe l'oeil mastery",
        "water splash frozen, liquid sculpture, hyper-realistic transparency",
    ],
    "expressionist_emotion": [
        "the scream bridge scene, munch anguish, swirling sky, existential",
        "bridge at night, kirchner style, sharp angles, emotional color",
        "portrait with distorted features, emotional not physical truth",
        "stormy landscape, kokoschka energy, turbulent brushwork, feeling",
        "figures in motion, egon schiele line, psychological intensity",
        "city street perspective warped, emotional impact through distortion",
        "self portrait in mirror, harsh light, confrontational honesty",
        "dancers with exaggerated movement, emotion through form and color",
    ],
    "op_art_illusion": [
        "black and white stripes bending, bridget riley optical vibration",
        "concentric circles creating movement, vasarely kinetic illusion",
        "geometric pattern shifting, anuszkiewicz color interaction",
        "impossible stairs, escher influence, perpetual motion illusion",
        "spirals creating depth, optical dizzy effect, perceptual puzzle",
        "parallel lines appearing curved, optical science as art",
        "checkerboard perspective distortion, space warping visually",
        "moire pattern interference, mathematical beauty, optical art",
    ],
    "mixed_media_texture": [
        "collage with fabric and paper, robert rauschenberg combine, textural",
        "assemblage with found objects, three dimensional surface, contemporary",
        "plaster and paint layers, anselm kiefer texture, material depth",
        "newspaper and acrylic, jasper johns technique, layers of meaning",
        "string and nails on board, geometric precision, tactile art",
        "sand and glue texture, beach memory, sculptural painting surface",
        "metallic leaf and oil paint, gustav klimt gold, decorative richness",
        "torn paper revealing layers, archaeological art, stratified history",
    ],
    "symbolist_mystery": [
        "woman with closed eyes surrounded by flowers, redon dream state",
        "angel with downcast gaze, burne-jones melancholy, medieval revival",
        "death and the maiden, symbolist allegory, memento mori beauty",
        "mysterious garden with hidden figures, symbolist narrative puzzle",
        "woman at mirror with double reflection, identity symbolism",
        "knight in dark forest, quest metaphor, symbolic journey",
        "sphinx riddle scene, enigmatic symbolism, philosophical mystery",
        "peacock displaying meaning, luxury and vanity symbol, decorative",
    ],
    "kinetic_movement": [
        "mobile sculpture in motion blur, calder influence, suspended balance",
        "spinning optical disc, motion creating color, kinetic illusion",
        "pendulum drawing pattern, harmonograph art, physics visualization",
        "rotating geometric forms, constructivist motion, mechanical ballet",
        "wind-activated sculpture, movement captured, environmental art",
        "light trails long exposure, motion painting photography, temporal art",
        "flip book animation single frame, motion suggested, sequential art",
        "zoetrope image slice, pre-cinema animation, movement captured",
    ],
    "organic_biomorphic": [
        "cellular forms multiplying, biomorphic abstraction, life force",
        "amorphous shapes flowing, jean arp organic sculpture influence",
        "coral-like structures branching, natural growth patterns as art",
        "embryonic forms suggesting life, organic abstraction, primordial",
        "plant-like tendrils reaching, art nouveau meets cellular biology",
        "bone structure abstract, organic architecture, natural engineering",
        "sea creature inspired forms, underwater biomorphism, fluid shapes",
        "microscopic life enlarged, scientific illustration as abstract art",
    ],
    "naive_folk_art": [
        "village scene with flat perspective, grandma moses americana",
        "animals in paradise garden, henri rousseau jungle dream, innocent eye",
        "circus performers, naive charm, bright colors, childlike wonder",
        "harvest celebration, folk art tradition, community gathering depicted",
        "memory painting of childhood, self-taught artist vision, authentic",
        "religious scene with gold background, icon painting influence, devotional",
        "market day busy scene, narrative folk art, storytelling in paint",
        "seasons represented symbolically, folk tradition, decorative border",
    ],
    "constructivist_design": [
        "red wedge composition, el lissitzky revolutionary, geometric power",
        "photomontage with text, rodchenko technique, propaganda aesthetic",
        "diagonal composition dynamic, constructivist energy, forward motion",
        "industrial forms celebrating machine age, modernist optimism",
        "typography as image, constructivist poster, bold sans serif",
        "geometric abstraction with purpose, form serving function",
        "scaffolding and cranes, construction as subject, industrial beauty",
        "suprematist composition, black square on white, ultimate abstraction",
    ],
    "symbolist_dreams": [
        "island of the dead, bocklin mystery, cypress trees, somber journey",
        "ophelia floating in stream, millais tragedy, pre-raphaelite detail",
        "lady of shalott in boat, waterhouse romance, arthurian legend",
        "pandora's box opening, symbolist narrative, mythology reimagined",
        "salome with head of john, beardsley decadence, art nouveau line",
        "circe offering cup, waterhouse enchantress, dangerous beauty",
        "persephone in underworld, seasonal myth, symbolist interpretation",
        "medusa reflection in shield, clever symbolism, indirect viewing",
    ],
    "color_field_meditation": [
        "horizontal bands of color, newman zip, contemplative space",
        "stained canvas soft edges, frankenthaler soak-stain, lyrical",
        "pure color rectangle, rothko spiritual, emotional chromatics",
        "color transition gradient, atmospheric space, boundless field",
        "monochrome with subtle variation, meditation on single hue",
        "color blocking large scale, immersive chromatic experience",
        "soft edge hard edge contrast, color temperature study",
        "luminous color glow, light emanating, transcendent hue",
    ],
    "assemblage_sculpture": [
        "nailed wood construction, louise nevelson wall, monochrome power",
        "bicycle parts welded, picasso bull's head wit, found object genius",
        "driftwood arrangement, natural forms assembled, organic sculpture",
        "metal scrap composition, industrial remnants, urban archaeology",
        "vintage tools organized, joseph cornell box precision, nostalgia",
        "typewriter keys collage, obsolete technology art, mechanical beauty",
        "clock parts gears exposed, steampunk assemblage, time deconstructed",
        "broken ceramics reconstructed, kintsugi philosophy, beauty in repair",
    ],
    "action_gesture": [
        "splattered paint energy, pollock drip technique, kinetic creation",
        "brushstroke calligraphic, franz kline gesture, black on white power",
        "paint flung with force, de kooning violence, controlled chaos",
        "one continuous line drawing, picasso without lifting pencil, pure gesture",
        "body paint performance, yves klein blue, living brushes concept",
        "quick sketch capturing essence, gestural economy, energy over detail",
        "finger painting primal, direct touch, unmediated expression",
        "automatic drawing, surrealist technique, unconscious gesture",
    ],
    "figurative_expression": [
        "seated figure heavy brush, freud psychological, flesh rendered thick",
        "bacon screaming pope, distorted portrait, existential horror",
        "hockney pool scene, bright california, acrylic clarity, figures relaxed",
        "jenny saville flesh study, monumental bodies, paint as flesh",
        "figurative in landscape, figurative expressionism, emotional narrative",
        "diebenkorn bay area figure, planes of color, west coast light",
        "swimmer underwater, chlorine blue, aquatic figure distortion",
        "conversation between figures, social realism, everyday moment elevated",
    ],
    "decorative_pattern": [
        "william morris wallpaper, intertwined vines, arts and crafts movement",
        "klimt portrait with patterns, gold leaf, decorative brilliance",
        "matisse cutouts, paper collage, joyful pattern and color",
        "textile design repeating motif, fabric art, rhythmic decoration",
        "islamic geometric pattern, mathematical beauty, spiritual decoration",
        "aboriginal dot painting, dreamtime story, symbolic pattern language",
        "paisley swirl elaborate, indian textile influence, organic pattern",
        "art deco repeated forms, luxury pattern, geometric elegance",
    ],
    "light_space": [
        "james turrell skyspace, aperture to sky, light as medium",
        "dan flavin fluorescent tubes, minimal light sculpture, color space",
        "olafur eliasson sun, artificial nature, atmospheric light installation",
        "light through prism, newton spectrum, rainbow cast in space",
        "gobo projected patterns, theatrical light, shadow play artistry",
        "fiber optic light drawing, technology as brush, illuminated gesture",
        "sunset color study, natural light phenomenon, atmospheric optics",
        "candlelight chiaroscuro, rembrandt lighting, intimate illumination",
    ],
    "social_commentary": [
        "guernica war horror, picasso protest, cubist tragedy, anti-war",
        "migrant mother photograph, dorothea lange, depression era, dignity",
        "american gothic satire, regionalism, farmer portrait, midwestern",
        "factory workers, rivera mural style, socialist realism, labor dignity",
        "protest signs and crowds, documentary style, social movement captured",
        "inequality visualization, economic disparity, contemporary social art",
        "environmental destruction, political landscape, earth as victim",
        "surveillance society, cameras watching, privacy erosion visualized",
    ],
}


def count_all_prompts() -> int:
    """Count total available prompts."""
    return sum(len(prompts) for prompts in ARTISTIC_PROMPTS_2.values())


def collect_prompts(categories: list = None, randomize: bool = False):
    """Collect prompts from specified categories."""
    available_prompts = []

    # Filter categories if specified
    prompt_dict = ARTISTIC_PROMPTS_2
    if categories:
        prompt_dict = {k: v for k, v in ARTISTIC_PROMPTS_2.items() if k in categories}

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
    print("🎨 AI ARTIST - ARTISTIC STYLES COLLECTION II")
    print(f"{'=' * 70}")
    print("📊 Statistics:")
    print(f"   • Total categories: {len(ARTISTIC_PROMPTS_2)}")
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


def generate_artistic_collection_2(
    num_images: int = None,
    categories: list = None,
    randomize: bool = True,
    vary_parameters: bool = True,
):
    """Generate second artistic collection with more style-focused prompts.

    Args:
        num_images: Number of images to generate (None = all prompts)
        categories: Specific categories to use (None = all categories)
        randomize: Whether to shuffle prompts randomly
        vary_parameters: Whether to vary generation parameters for variety
    """
    logger.info(
        "artistic_collection_2_started",
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
                    "collection": "artistic_2",
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
        "artistic_collection_2_complete",
        success=success_count,
        failed=len(failed_prompts),
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate second artistic collection with more creative styles"
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
        print("\n📋 Available Artistic Categories (Collection 2):\n")
        for i, (category, prompts) in enumerate(ARTISTIC_PROMPTS_2.items(), 1):
            print(f"{i:2}. {category:30} ({len(prompts)} prompts)")
        print(
            f"\nTotal: {len(ARTISTIC_PROMPTS_2)} categories, {count_all_prompts()} prompts\n"
        )
        return

    generate_artistic_collection_2(
        num_images=args.num_images,
        categories=args.categories,
        randomize=not args.no_randomize,
        vary_parameters=not args.no_vary,
    )


if __name__ == "__main__":
    main()
