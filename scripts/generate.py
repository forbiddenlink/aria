#!/usr/bin/env python3
"""Universal image generation script for AI Artist.

Consolidates all generation modes into one easy-to-use interface.
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_artist.core.generator import ImageGenerator
from ai_artist.utils.config import load_config
from ai_artist.utils.logging import get_logger

logger = get_logger(__name__)


# Prompt collections for different modes
CREATIVE_PROMPTS = [
    "ethereal cosmic landscape with swirling nebulas",
    "abstract geometric patterns in vibrant colors",
    "surreal dreamscape with floating islands",
    "bioluminescent forest at midnight",
    "crystalline structures in an alien world",
]

DIVERSE_PROMPTS = [
    "majestic mountain landscape at golden hour",
    "cozy reading nook with warm lighting",
    "futuristic cityscape at night",
    "serene japanese garden with cherry blossoms",
    "vintage cafe interior with afternoon sunlight",
]

NATURE_PROMPTS = [
    "dramatic ocean waves crashing on rocky shore",
    "misty forest with rays of sunlight filtering through",
    "vibrant autumn foliage in mountain valley",
    "pristine alpine lake with mountain reflections",
    "desert sand dunes at sunset",
]

ULTIMATE_PROMPTS = [
    # Landscapes
    "epic mountain vista with dramatic clouds",
    "tropical paradise beach at sunset",
    "northern lights over snowy landscape",
    # Abstract
    "flowing liquid colors in motion",
    "geometric sacred geometry patterns",
    "cosmic energy waves",
    # Urban
    "cyberpunk city street at night with neon",
    "cozy european cobblestone street",
    "modern minimalist architecture",
    # Fantasy
    "mystical forest with glowing mushrooms",
    "ancient temple ruins overgrown with vines",
    "ethereal crystal cave",
]


def generate_single(config: dict, prompt: str = None, output_dir: Path = None):
    """Generate a single image."""
    if not prompt:
        prompt = input("Enter prompt: ").strip()
        if not prompt:
            print("❌ Prompt cannot be empty")
            return
    
    print(f"\n🎨 Generating: {prompt}")
    
    generator = ImageGenerator(config)
    images = generator.generate(
        prompt=prompt,
        num_images=1,
        output_dir=output_dir or Path("gallery")
    )
    
    if images:
        print(f"✅ Generated: {images[0]}")
    else:
        print("❌ Generation failed")


def generate_batch(config: dict, num_images: int, output_dir: Path = None):
    """Generate multiple images from user prompts."""
    prompts = []
    
    print(f"\n📝 Enter {num_images} prompts (or press Enter to use defaults):")
    for i in range(num_images):
        prompt = input(f"  Prompt {i+1}/{num_images}: ").strip()
        if prompt:
            prompts.append(prompt)
        else:
            prompts.append(f"beautiful artistic scene {i+1}")
    
    generator = ImageGenerator(config)
    print(f"\n🎨 Generating {len(prompts)} images...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt}")
        images = generator.generate(
            prompt=prompt,
            num_images=1,
            output_dir=output_dir or Path("gallery")
        )
        if images:
            print(f"  ✅ {images[0]}")
        else:
            print("  ❌ Failed")


def generate_collection(
    config: dict,
    prompts: List[str],
    collection_name: str,
    output_dir: Path = None
):
    """Generate a themed collection."""
    print(f"\n🎨 Generating {collection_name} Collection ({len(prompts)} images)")
    print("=" * 70)
    
    generator = ImageGenerator(config)
    output_path = output_dir or Path("gallery")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt}")
        images = generator.generate(
            prompt=prompt,
            num_images=1,
            output_dir=output_path
        )
        if images:
            print(f"  ✅ {images[0].name}")
        else:
            print("  ❌ Failed")
    
    print(f"\n✅ {collection_name} collection complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Universal image generation for AI Artist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with prompt
  python scripts/generate.py --mode single --prompt "mountain landscape"
  
  # Interactive single image
  python scripts/generate.py --mode single
  
  # Generate 5 images
  python scripts/generate.py --mode batch --num 5
  
  # Creative collection
  python scripts/generate.py --mode creative
  
  # Diverse collection
  python scripts/generate.py --mode diverse
  
  # Ultimate collection (12 images)
  python scripts/generate.py --mode ultimate
  
  # Custom prompts from file
  python scripts/generate.py --mode custom --prompts-file my_prompts.txt
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "creative", "diverse", "nature", "ultimate", "custom"],
        default="single",
        help="Generation mode"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for single mode"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of images for batch mode"
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="File with prompts (one per line) for custom mode"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: gallery/)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Config file path"
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1
    
    print("🎨 AI Artist - Universal Generator")
    print("=" * 70)
    
    # Check for LoRA
    lora_path = config.get("model", {}).get("lora_path")
    lora_scale = config.get("model", {}).get("lora_scale", 0.8)
    if lora_path:
        lora_name = Path(lora_path).name
        print(f"✨ LoRA Active: {lora_name} (scale: {lora_scale})")
    else:
        print("✨ Base Model (no LoRA)")
    print()
    
    try:
        # Route to appropriate handler
        if args.mode == "single":
            generate_single(config, args.prompt, args.output_dir)
        
        elif args.mode == "batch":
            generate_batch(config, args.num, args.output_dir)
        
        elif args.mode == "creative":
            generate_collection(config, CREATIVE_PROMPTS, "Creative", args.output_dir)
        
        elif args.mode == "diverse":
            generate_collection(config, DIVERSE_PROMPTS, "Diverse", args.output_dir)
        
        elif args.mode == "nature":
            generate_collection(config, NATURE_PROMPTS, "Nature", args.output_dir)
        
        elif args.mode == "ultimate":
            generate_collection(config, ULTIMATE_PROMPTS, "Ultimate", args.output_dir)
        
        elif args.mode == "custom":
            if not args.prompts_file or not args.prompts_file.exists():
                print("❌ --prompts-file required for custom mode")
                return 1
            
            prompts = args.prompts_file.read_text().strip().split("\n")
            prompts = [p.strip() for p in prompts if p.strip()]
            
            if not prompts:
                print("❌ No prompts found in file")
                return 1
            
            generate_collection(config, prompts, "Custom", args.output_dir)
        
        print("\n✨ Generation complete!")
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user")
        return 130
    except Exception as e:
        logger.exception("generation_failed", error=str(e))
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
