#!/usr/bin/env python3
"""Generate an artistic collection focused on creative styles and abstract concepts.

Third collection featuring artistic movements, abstract visualizations, and creative styles.
"""

import sys
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_artist.main import AIArtist
from src.ai_artist.utils.logging import configure_logging, get_logger
from src.ai_artist.utils.config import load_config

configure_logging()
logger = get_logger(__name__)

# Artistic and abstract prompt collection
ARTISTIC_PROMPTS = {
    "impressionist_dreams": [
        "water lilies pond, monet style, soft brushstrokes, pastel colors, dreamy atmosphere",
        "parisian cafe terrace at night, warm glowing lights, loose brushwork, impressionist",
        "hay stacks at different times of day, golden light, impressionist landscape",
        "ballet dancers backstage, degas style, soft pastels, movement captured",
        "river seine with boat reflections, renoir inspired, dappled light, romantic",
        "garden party in summer, parasols and dresses, impressionist technique",
        "wheat field with poppies, vibrant reds and golds, textured brushstrokes",
        "bridge over pond, japanese influence, atmospheric perspective, impressionist",
    ],
    
    "abstract_expressionism": [
        "color field explosion, rothko inspired, layers of emotion, bold rectangles",
        "action painting, pollock style, energetic drips and splatters, controlled chaos",
        "geometric abstraction, mondrian influence, primary colors, black lines",
        "abstract landscape suggestion, kandinsky style, musical composition in color",
        "biomorphic forms floating, miro inspired, playful abstract shapes",
        "color blocks in tension, malevich suprematism, pure geometric feeling",
        "gestural brushstrokes, de kooning energy, aggressive yet graceful marks",
        "meditative color gradients, newman style, vertical zips, contemplative",
    ],
    
    "surrealist_visions": [
        "persistence of time, soft watches draped, desert landscape, dali inspired",
        "floating green apple obscuring face, magritte mystery, suit and bowler hat",
        "elephant with spider legs, fragile yet massive, surreal contradiction",
        "keyhole revealing eye, looking through layers, voyeuristic surrealism",
        "chess pieces on infinite board, strategy and futility, metaphysical",
        "double image optical illusion, two scenes in one, perceptual puzzle",
        "burning giraffe, freudian symbolism, war and destruction, surreal nightmare",
        "lovers with cloth-covered faces, intimacy and anonymity, romantic mystery",
    ],
    
    "cubist_deconstruction": [
        "portrait from multiple angles simultaneously, picasso inspired, fractured planes",
        "guitar and bottle on table, braque style, muted colors, overlapping shapes",
        "cityscape fragmented into geometric forms, sharp angles, analytical cubism",
        "still life with newspaper, collage elements, synthetic cubism technique",
        "violin deconstructed, musical instrument as abstract form, multiple perspectives",
        "figure descending staircase, motion through cubist lens, duchamp influence",
        "three musicians, flat colored shapes, synthetic cubism, rhythmic composition",
        "african mask influence, primitive and modern merged, cubist portrait",
    ],
    
    "art_nouveau_elegance": [
        "flowing hair becoming floral vines, mucha style, decorative border, organic",
        "peacock feather patterns, iridescent colors, art nouveau curves",
        "woman in flowing dress, botanical elements, poster design, ornamental",
        "stained glass with nature motifs, lead lines, jewel tones, architectural art nouveau",
        "dragonfly wing details, geometric yet organic, jewelry design aesthetic",
        "whiplash curves and tendrils, klimt gold influence, decorative patterns",
        "four seasons personified, allegorical figures, art nouveau symbolism",
        "iris flowers stylized, purple and green harmony, poster art elegance",
    ],
    
    "pop_art_bold": [
        "comic book explosion, roy lichtenstein dots, bold outlines, primary colors",
        "soup can array, warhol repetition, consumer culture commentary",
        "celebrity portrait screen print, vibrant contrasts, iconic pop imagery",
        "collage of magazine cutouts, david hockney pool, californian bright",
        "comic speech bubble, 'wow!', benday dots, graphic novel aesthetic",
        "enlarged everyday object, claes oldenburg scale, mundane made monumental",
        "british flag color palette, peter blake collage, nostalgic assemblage",
        "halftone portrait, multiple color variations, repetition and variation",
    ],
    
    "minimalist_zen": [
        "single line drawing, continuous contour, elegant simplicity, less is more",
        "three stones stacked, perfect balance, negative space, meditative",
        "square within square, albers homage, subtle color relationships",
        "empty white canvas with tiny dot, contemplation of nothing and everything",
        "parallel lines in space, repetition and rhythm, optical subtlety",
        "monochrome gradient, smooth transition, atmospheric minimalism",
        "grid system, sol lewitt mathematical, conceptual minimalism",
        "light and shadow only, no objects, architectural minimalism, pure form",
    ],
    
    "street_art_urban": [
        "colorful graffiti mural, hip hop culture, spray paint texture, urban energy",
        "banksy style political stencil, ironic message, guerrilla art",
        "yarn bombing tree, knitted street art, colorful textile installation",
        "wheatpaste poster layers, weathered urban texture, contemporary folk art",
        "3d anamorphic street painting, optical illusion on pavement, interactive",
        "mosaic portrait from tiles, invader pixel art, video game aesthetic",
        "stencil portrait layers, multiple colors, street art technique",
        "calligraffiti fusion, arabic script meets graffiti, cultural blend",
    ],
    
    "digital_glitch_art": [
        "pixel sorting cascade, data moshing, corrupted image beauty",
        "rgb color channel separation, chromatic aberration, glitch aesthetic",
        "broken jpeg artifacts, compression errors as art, digital decay",
        "datamosh face portrait, temporal distortion, video glitch frozen",
        "ascii art portrait, text characters forming image, retro computer",
        "circuit bent colors, hardware malfunction aesthetic, beautiful errors",
        "vhs tracking errors, analog glitch, nostalgic distortion",
        "database corruption visualization, numbers becoming abstract art",
    ],
    
    "watercolor_fluidity": [
        "loose floral bouquet, wet-on-wet technique, colors bleeding beautifully",
        "misty mountain landscape, atmospheric watercolor, soft edges, ethereal",
        "abstract wash layers, transparent overlaps, organic color mixing",
        "portrait with dripping colors, emotional watercolor, expressive looseness",
        "ocean waves, fluid motion captured, turquoise washes, coastal scene",
        "autumn trees, warm washes, negative space, leaves suggested not detailed",
        "rainy city street, reflections and atmosphere, moody watercolor",
        "koi pond from above, fish in water, depth through transparency",
    ],
    
    "art_deco_geometry": [
        "chrysler building spire, stepped geometric forms, metallic art deco",
        "jazz age poster, geometric figures dancing, gold and black elegance",
        "sunburst pattern radiating, symmetrical art deco motif, streamline moderne",
        "egyptian revival design, geometric hieroglyphs, 1920s glamour",
        "ziggurat stepped pyramid, art deco architecture, bold angular forms",
        "stylized gazelle leaping, art deco animal, elegant linear forms",
        "fan pattern ceiling, geometric luxury, art deco interior detail",
        "speed lines and chrome, machine age aesthetic, streamlined forms",
    ],
    
    "psychedelic_experience": [
        "concentric rainbow circles expanding, optical vibration, trippy patterns",
        "paisley swirls and fractals, 1960s poster art, mind-expanding visuals",
        "melting face morphing, liquid reality, psychedelic transformation",
        "kaleidoscope mandala, infinite symmetry, meditation visual",
        "warping checkerboard floor, perspective distortion, reality bending",
        "fluorescent colors under blacklight, day-glo intensity, rave aesthetic",
        "eye with spiral pupil, hypnotic gaze, consciousness visualization",
        "mushroom forest glowing, alice in wonderland trip, magical realism",
    ],
    
    "chiaroscuro_drama": [
        "single candle illuminating face, caravaggio lighting, dramatic shadows",
        "fruits on table, renaissance still life, raking light, dutch master style",
        "figure emerging from darkness, rembrandt technique, spotlight effect",
        "hands in prayer, tenebrism, spiritual light from above, baroque drama",
        "musician in shadow playing instrument, vermeer light quality, intimate",
        "elderly person reading by window, soft directional light, contemplative",
        "wine glass and bread, last supper lighting, symbolic still life",
        "self portrait half in shadow, psychological depth, dramatic contrast",
    ],
    
    "collage_assemblage": [
        "vintage magazine cutouts layered, retro mixed media, nostalgic compilation",
        "found objects arranged, joseph cornell box, shadow box art",
        "torn paper pieces, abstract composition, textured layers",
        "photographs and text combined, dada influence, nonsense poetry",
        "fabric swatches quilted, textile collage, pattern on pattern",
        "botanical specimens pressed, scientific meets artistic, herbarium art",
        "ticket stubs and ephemera, memory collage, time capsule aesthetic",
        "maps torn and layered, geographical abstraction, journey compilation",
    ],
    
    "ink_illustration": [
        "detailed pen and ink cityscape, crosshatching technique, architectural drawing",
        "botanical illustration, scientific accuracy, stippling texture",
        "portrait with fine lines, engraving style, classical technique",
        "japanese sumi-e bamboo, minimal brush strokes, zen simplicity",
        "comic book noir scene, heavy blacks, dramatic ink work",
        "calligraphy forming image, letterforms as art, typographic illustration",
        "woodcut print aesthetic, bold black and white, folk art simplicity",
        "ink wash landscape, chinese mountain painting, misty atmospheric",
    ],
    
    "fauvism_wild_color": [
        "landscape in unnatural colors, matisse inspired, decorative flatness",
        "portrait with green face, fauvist palette, emotional color choice",
        "harbor with boats, bright unrealistic hues, wild colorist approach",
        "interior with red room, pattern overload, fauvist decoration",
        "trees in purple and orange, liberated color, expressive not realistic",
        "still life with impossible colors, emotional temperature through hue",
        "dancers in vivid non-local color, movement through fauvism",
        "window view with clashing colors, matisse cutout influence",
    ],
    
    "gothic_romanticism": [
        "ruined abbey under moonlight, caspar david friedrich, sublime nature",
        "lone figure on mountain peak, romantic contemplation, infinite landscape",
        "shipwreck in storm, turner drama, nature's overwhelming power",
        "cemetery with weeping angels, gothic melancholy, marble and ivy",
        "haunted castle on cliff, dark romanticism, mysterious atmosphere",
        "ancient oak tree, gnarled and twisted, romantic symbolism",
        "fog rolling over moors, bronte landscape, lonely and beautiful",
        "medieval cathedral interior, gothic arches, spiritual light",
    ],
    
    "pointillism_dots": [
        "sunday afternoon park scene, seurat style, countless tiny dots",
        "circus performance, signac technique, optical color mixing",
        "harbor at sunset, pointillist dots creating luminous water",
        "portrait built from colored points, divisionism technique",
        "eiffel tower through trees, neo-impressionist dots, parisian view",
        "dancers in dotted light, chromoluminarism, vibrant optical mixing",
        "beach scene with parasols, summer light through pointillism",
        "garden path, dot by dot creation, patient technique, luminous result",
    ],
    
    "bauhaus_functional": [
        "geometric furniture design, form follows function, primary colors",
        "typography poster, sans serif boldness, bauhaus grid system",
        "architectural blueprint aesthetic, clean lines, modernist vision",
        "abstract shapes teaching color theory, itten influence, systematic",
        "steel tube chair, marcel breuer style, industrial materials as art",
        "workshop tools arranged, function as beauty, honest materials",
        "modular design system, repeating elements, bauhaus efficiency",
        "light and shadow study, moholy-nagy photogram, experimental",
    ],
    
    "magical_realism": [
        "woman with butterflies emerging from book, reality with magic touch",
        "house floating on cloud, dreamlike yet painted realistically",
        "tree growing books as leaves, literacy metaphor, surreal but detailed",
        "hourglass containing miniature seasons changing, time visualization",
        "umbrella raining upward, impossibility rendered realistically",
        "staircase leading underwater seamlessly, two worlds merged naturally",
        "girl with galaxy hair, cosmos in portrait, magical yet grounded",
        "violin releasing birds as music, synesthesia visualized, poetic realism",
    ],
}


def count_all_prompts() -> int:
    """Count total available prompts."""
    return sum(len(prompts) for prompts in ARTISTIC_PROMPTS.values())


def collect_prompts(categories: list = None, randomize: bool = False):
    """Collect prompts from specified categories."""
    available_prompts = []
    
    # Filter categories if specified
    prompt_dict = ARTISTIC_PROMPTS
    if categories:
        prompt_dict = {k: v for k, v in ARTISTIC_PROMPTS.items() if k in categories}
    
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
    print(f"\n{'='*70}")
    print(f"🎨 AI ARTIST - ARTISTIC STYLES COLLECTION")
    print(f"{'='*70}")
    print(f"📊 Statistics:")
    print(f"   • Total categories: {len(ARTISTIC_PROMPTS)}")
    print(f"   • Available prompts: {count_all_prompts()}")
    print(f"   • Generating: {total} artworks")
    print(f"   • Device: {artist.generator.device}")
    print(f"   • Model: {artist.generator.model_id}")
    print(f"   • Randomized: {'Yes' if randomize else 'No'}")
    print(f"   • Parameter variation: {'Yes' if vary_parameters else 'No'}")
    print(f"{'='*70}\n")


def print_progress(i: int, total: int, category: str, prompt: str, params: dict, vary: bool):
    """Print progress for current generation."""
    print(f"[{i}/{total}] Category: {category}")
    print(f"   Prompt: {prompt[:65]}...")
    if vary:
        print(f"   Params: steps={params['steps']}, guidance={params['guidance']}, seed={params['seed']}")


def print_summary(success_count: int, total: int, failed_prompts: list, category_counts: dict):
    """Print final summary."""
    print(f"\n{'='*70}")
    print(f"✨ GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 Results:")
    print(f"   ✅ Successful: {success_count}/{total} ({success_count/total*100:.1f}%)")
    print(f"   ❌ Failed: {len(failed_prompts)}")
    print(f"\n📁 Category breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {cat}: {count} images")
    
    if failed_prompts:
        print(f"\n❌ Failed prompts:")
        for cat, prompt in failed_prompts:
            print(f"   • [{cat}] {prompt[:50]}...")
    
    print(f"\n💾 All images saved to gallery/2026/")
    print(f"{'='*70}\n")


def generate_artistic_collection(
    num_images: int = None,
    categories: list = None,
    randomize: bool = True,
    vary_parameters: bool = True,
):
    """Generate artistic collection with style-focused prompts.
    
    Args:
        num_images: Number of images to generate (None = all prompts)
        categories: Specific categories to use (None = all categories)
        randomize: Whether to shuffle prompts randomly
        vary_parameters: Whether to vary generation parameters for variety
    """
    logger.info("artistic_collection_started", 
                num_images=num_images, 
                categories=categories,
                randomize=randomize)
    
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
                    "collection": "artistic",
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
                print(f"   ❌ Generation failed\n")
                
        except KeyboardInterrupt:
            logger.info("generation_interrupted_by_user")
            print("\n\n⚠️  Generation interrupted by user")
            break
        except Exception as e:
            logger.error("generation_failed", category=category, prompt=prompt[:50], error=str(e))
            failed_prompts.append((category, prompt))
            print(f"   ❌ Error: {e}\n")
            continue
    
    # Print final summary
    print_summary(success_count, total, failed_prompts, category_counts)
    
    # Cleanup
    artist.generator.unload()
    logger.info("artistic_collection_complete", 
                success=success_count, 
                failed=len(failed_prompts))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate artistic collection with style-focused prompts"
    )
    parser.add_argument(
        "-n", "--num-images",
        type=int,
        help="Number of images to generate (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all prompts (same as omitting -n)",
    )
    parser.add_argument(
        "-c", "--categories",
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
        print("\n📋 Available Artistic Categories:\n")
        for i, (category, prompts) in enumerate(ARTISTIC_PROMPTS.items(), 1):
            print(f"{i:2}. {category:30} ({len(prompts)} prompts)")
        print(f"\nTotal: {len(ARTISTIC_PROMPTS)} categories, {count_all_prompts()} prompts\n")
        return
    
    generate_artistic_collection(
        num_images=args.num_images,
        categories=args.categories,
        randomize=not args.no_randomize,
        vary_parameters=not args.no_vary,
    )


if __name__ == "__main__":
    main()
