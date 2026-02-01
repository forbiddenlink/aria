#!/usr/bin/env python3
"""Generate artwork collections using various prompt collections.

Unified script that replaces the separate generation scripts:
- generate_artistic_collection.py
- generate_artistic_collection_2.py
- generate_expanded_collection.py
- generate_ultimate_collection.py

Usage:
    # Generate 10 images from artistic collection
    python scripts/generate_collection.py --collection artistic -n 10

    # Generate all prompts from ultimate collection
    python scripts/generate_collection.py --collection ultimate --all

    # Generate from all collections combined
    python scripts/generate_collection.py --collection all -n 50

    # List categories in a collection
    python scripts/generate_collection.py --collection expanded --list-categories

    # Generate specific categories only
    python scripts/generate_collection.py --collection ultimate -c cosmic_wonders mythological_beings -n 20
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_artist.main import AIArtist
from src.ai_artist.prompts import (
    count_prompts,
    get_collection_info,
    get_collection_names,
    get_collection_prompts,
)
from src.ai_artist.utils.config import load_config
from src.ai_artist.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def collect_prompts(
    collection_name: str,
    categories: list[str] | None = None,
    randomize: bool = False,
) -> list[tuple[str, str]]:
    """Collect prompts from specified collection and categories.

    Args:
        collection_name: Name of the collection to use
        categories: Specific categories to filter (None = all)
        randomize: Whether to shuffle the prompts

    Returns:
        List of (category, prompt) tuples
    """
    prompts_dict = get_collection_prompts(collection_name)

    # Filter categories if specified
    if categories:
        prompts_dict = {k: v for k, v in prompts_dict.items() if k in categories}

    # Collect all prompts with their categories
    available_prompts = []
    for category, prompts in prompts_dict.items():
        for prompt in prompts:
            available_prompts.append((category, prompt))

    # Randomize if requested
    if randomize:
        random.shuffle(available_prompts)

    return available_prompts


def get_generation_params(vary_parameters: bool) -> dict:
    """Get generation parameters with optional variation.

    Args:
        vary_parameters: Whether to vary parameters for variety

    Returns:
        Dictionary with steps, guidance, and seed
    """
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


def print_header(
    collection_name: str,
    total: int,
    artist: "AIArtist",
    randomize: bool,
    vary_parameters: bool,
):
    """Print generation header with collection info."""
    info = get_collection_info(collection_name) if collection_name != "all" else None
    collection_display = info["name"] if info else "ALL COLLECTIONS"

    print(f"\n{'=' * 70}")
    print(f"AI ARTIST - {collection_display}")
    print(f"{'=' * 70}")
    print("Statistics:")

    if collection_name == "all":
        total_categories = sum(
            get_collection_info(c)["categories"] for c in get_collection_names()
        )
        total_prompts = count_prompts()
        print(f"   Total collections: {len(get_collection_names())}")
        print(f"   Total categories: {total_categories}")
        print(f"   Available prompts: {total_prompts}")
    else:
        print(f"   Total categories: {info['categories']}")
        print(f"   Available prompts: {info['total_prompts']}")

    print(f"   Generating: {total} artworks")
    print(f"   Device: {artist.generator.device}")
    print(f"   Model: {artist.generator.model_id}")
    print(f"   Randomized: {'Yes' if randomize else 'No'}")
    print(f"   Parameter variation: {'Yes' if vary_parameters else 'No'}")
    print(f"{'=' * 70}\n")


def print_progress(
    i: int,
    total: int,
    category: str,
    prompt: str,
    params: dict,
    vary: bool,
):
    """Print progress for current generation."""
    print(f"[{i}/{total}] Category: {category}")
    print(f"   Prompt: {prompt[:65]}...")
    if vary:
        print(
            f"   Params: steps={params['steps']}, "
            f"guidance={params['guidance']}, seed={params['seed']}"
        )


def print_summary(
    success_count: int,
    total: int,
    failed_prompts: list,
    category_counts: dict,
):
    """Print final generation summary."""
    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE!")
    print(f"{'=' * 70}")
    print("Results:")
    if total > 0:
        print(
            f"   Successful: {success_count}/{total} ({success_count / total * 100:.1f}%)"
        )
    else:
        print(f"   Successful: {success_count}/{total}")
    print(f"   Failed: {len(failed_prompts)}")
    print("\nCategory breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat}: {count} images")

    if failed_prompts:
        print("\nFailed prompts:")
        for item in failed_prompts[:10]:
            if len(item) == 3:
                idx, cat, prompt = item
                print(f"   [{idx}] {cat}: {prompt[:50]}...")
            else:
                cat, prompt = item
                print(f"   [{cat}] {prompt[:50]}...")
        if len(failed_prompts) > 10:
            print(f"   ... and {len(failed_prompts) - 10} more")

    print("\nAll images saved to gallery/")
    print(f"{'=' * 70}\n")


def generate_collection(
    collection_name: str,
    num_images: int | None = None,
    categories: list[str] | None = None,
    randomize: bool = True,
    vary_parameters: bool = True,
):
    """Generate artwork from specified collection.

    Args:
        collection_name: Collection to use ('artistic', 'artistic2', 'expanded', 'ultimate', 'all')
        num_images: Number of images to generate (None = all prompts)
        categories: Specific categories to use (None = all categories)
        randomize: Whether to shuffle prompts randomly
        vary_parameters: Whether to vary generation parameters for variety
    """
    logger.info(
        "collection_generation_started",
        collection=collection_name,
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
    available_prompts = collect_prompts(collection_name, categories, randomize)

    # Limit to requested number
    if num_images:
        available_prompts = available_prompts[:num_images]

    total = len(available_prompts)

    # Display header
    print_header(collection_name, total, artist, randomize, vary_parameters)

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
                    "collection": collection_name,
                }

                saved_path = artist.gallery.save_image(
                    image=images[0],
                    prompt=prompt,
                    metadata=metadata,
                    featured=False,
                )

                success_count += 1
                print(f"   Saved: {saved_path.name}\n")
            else:
                failed_prompts.append((i, category, prompt))
                print("   Generation failed\n")

        except KeyboardInterrupt:
            logger.info("generation_interrupted_by_user")
            print("\n\nGeneration interrupted by user")
            break
        except Exception as e:
            logger.error(
                "generation_failed",
                category=category,
                prompt=prompt[:50],
                error=str(e),
            )
            failed_prompts.append((i, category, prompt))
            print(f"   Error: {e}\n")
            continue

    # Print final summary
    print_summary(success_count, total, failed_prompts, category_counts)

    # Show gallery location
    print("View your gallery:")
    print("   Web interface: http://localhost:8000")
    print(f"   File location: {artist.gallery.gallery_dir}")
    print(f"{'=' * 70}\n")

    # Cleanup
    artist.generator.unload()
    logger.info(
        "collection_generation_complete",
        collection=collection_name,
        success=success_count,
        failed=len(failed_prompts),
    )


def list_categories(collection_name: str):
    """List all categories in a collection."""
    prompts = get_collection_prompts(collection_name)

    print(f"\nAvailable Categories ({collection_name}):\n")
    for i, (category, prompt_list) in enumerate(sorted(prompts.items()), 1):
        print(f"{i:2}. {category:35} ({len(prompt_list)} prompts)")

    total_prompts = sum(len(p) for p in prompts.values())
    print(f"\nTotal: {len(prompts)} categories, {total_prompts} prompts\n")


def list_collections():
    """List all available collections."""
    print("\nAvailable Collections:\n")
    for name in get_collection_names():
        info = get_collection_info(name)
        print(f"  {name:12} - {info['name']}")
        print(
            f"               {info['categories']} categories, {info['total_prompts']} prompts"
        )
        print(f"               {info['description']}\n")
    print(f"  {'all':12} - All collections combined")
    print(f"               {count_prompts()} total prompts\n")


def main():
    """Main entry point."""
    # Get all valid collection names for argument validation
    valid_collections = get_collection_names() + ["all"]

    parser = argparse.ArgumentParser(
        description="Generate artwork collections from various prompt sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 images from artistic collection
  python scripts/generate_collection.py --collection artistic -n 10

  # Generate all prompts from ultimate collection
  python scripts/generate_collection.py --collection ultimate --all

  # Generate from all collections combined
  python scripts/generate_collection.py --collection all -n 50

  # List available collections
  python scripts/generate_collection.py --list-collections

  # List categories in a collection
  python scripts/generate_collection.py --collection expanded --list-categories

  # Generate specific categories only
  python scripts/generate_collection.py --collection ultimate -c cosmic_wonders mythological_beings
        """,
    )

    parser.add_argument(
        "--collection",
        type=str,
        choices=valid_collections,
        default="ultimate",
        help="Collection to generate from (default: ultimate)",
    )
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        help="Number of images to generate (default: 10, omit for all with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all prompts in the collection",
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
        help="List all available categories in the collection and exit",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all available collections and exit",
    )

    args = parser.parse_args()

    # Handle list operations
    if args.list_collections:
        list_collections()
        return

    if args.list_categories:
        list_categories(args.collection)
        return

    # Determine number of images
    num_images = None
    if not args.all:
        num_images = args.num_images or 10  # Default to 10 if not specified

    # Run generation
    generate_collection(
        collection_name=args.collection,
        num_images=num_images,
        categories=args.categories,
        randomize=not args.no_randomize,
        vary_parameters=not args.no_vary,
    )


if __name__ == "__main__":
    main()
