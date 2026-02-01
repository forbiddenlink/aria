#!/usr/bin/env python3
"""Download specialized training datasets for focused LoRA training."""

import argparse
import time
from pathlib import Path

import requests


def download_from_urls(urls: list, output_dir: Path, prefix: str):
    """Download images from a list of URLs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for idx, url in enumerate(urls, 1):
        try:
            response = requests.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()

            filename = f"{prefix}_{idx:03d}.jpg"
            filepath = output_dir / filename

            with open(filepath, "wb") as f:
                f.write(response.content)

            print(f"  ✓ Downloaded {idx}/{len(urls)}: {filename}")
            downloaded += 1
            time.sleep(0.5)

        except Exception as e:
            print(f"  ✗ Failed to download image {idx}: {e}")
            continue

    return downloaded


def download_abstract_art(output_dir: Path, num_images: int = 30):
    """Download abstract art images - colorful, artistic compositions."""
    print("🎨 Downloading Abstract Art dataset...")
    print("   Style: Modern abstract, colorful, artistic")
    print()

    # Use Picsum with specific seeds for abstract-looking images
    # These seeds were manually selected for abstract/colorful compositions
    seeds = [
        100,
        150,
        200,
        300,
        350,
        400,
        450,
        500,
        600,
        700,
        800,
        900,
        1100,
        1200,
        1300,
        1400,
        1500,
        1700,
        1800,
        1900,
        2100,
        2200,
        2300,
        2500,
        2700,
        2900,
        3100,
        3300,
        3500,
        3700,
        3900,
        4100,
        4300,
        4500,
        4700,
    ]

    urls = [f"https://picsum.photos/seed/{seed}/768/768" for seed in seeds[:num_images]]
    return download_from_urls(urls, output_dir, "abstract")


def download_landscape_photos(output_dir: Path, num_images: int = 30):
    """Download landscape photography - dramatic, natural scenes."""
    print("🏔️  Downloading Landscape Photography dataset...")
    print("   Style: Natural landscapes, dramatic scenery")
    print()

    # Picsum seeds selected for landscape-style images
    seeds = [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        110,
        120,
        130,
        140,
        160,
        170,
        180,
        190,
        210,
        220,
        230,
        240,
        250,
        260,
        270,
        280,
        290,
        310,
        320,
        330,
        340,
        360,
        370,
        380,
        390,
        410,
    ]

    urls = [
        f"https://picsum.photos/seed/{seed}/1024/768" for seed in seeds[:num_images]
    ]
    return download_from_urls(urls, output_dir, "landscape")


def download_web_hero_images(output_dir: Path, num_images: int = 30):
    """Download professional web hero style images - clean, modern, professional."""
    print("💼 Downloading Web Hero Images dataset...")
    print("   Style: Professional, modern, clean compositions")
    print("   Perfect for: Website headers, landing pages, marketing")
    print()

    # Picsum seeds for clean, professional, well-composed images
    # These work well for hero sections
    seeds = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
    ]

    # 16:9 aspect ratio - perfect for web heroes
    urls = [
        f"https://picsum.photos/seed/{seed}/1920/1080" for seed in seeds[:num_images]
    ]
    return download_from_urls(urls, output_dir, "webhero")


DATASET_TYPES = {
    "abstract": {
        "name": "Abstract Art",
        "function": download_abstract_art,
        "output": "datasets/training_abstract",
        "description": "Colorful, artistic, modern abstract compositions",
        "use_case": "Artistic designs, creative projects, abstract artwork",
        "training_params": "--rank 16 --max_train_steps 3000 --learning_rate 5e-5",
    },
    "landscape": {
        "name": "Landscape Photography",
        "function": download_landscape_photos,
        "output": "datasets/training_landscape",
        "description": "Natural landscapes, dramatic scenery, outdoor photography",
        "use_case": "Nature backgrounds, travel imagery, outdoor scenes",
        "training_params": "--rank 8 --max_train_steps 3000 --learning_rate 1e-4",
    },
    "webhero": {
        "name": "Web Hero Images",
        "function": download_web_hero_images,
        "output": "datasets/training_webhero",
        "description": "Professional, clean, modern compositions (16:9 format)",
        "use_case": "Website headers, landing pages, hero sections, marketing",
        "training_params": "--rank 8 --max_train_steps 2500 --learning_rate 1e-4 --resolution 768",
    },
}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download specialized training datasets for LoRA"
    )
    parser.add_argument(
        "dataset_type",
        choices=list(DATASET_TYPES.keys()) + ["all"],
        help="Type of dataset to download",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=30,
        help="Number of images to download per dataset",
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory")

    args = parser.parse_args()

    print("🎨 AI Artist - Specialized Dataset Downloader")
    print("=" * 70)
    print()

    if args.dataset_type == "all":
        print("📦 Downloading ALL datasets...")
        print()

        for _dtype, info in DATASET_TYPES.items():
            output_dir = Path(info["output"])
            print(f"Dataset: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Output: {output_dir}")
            print()

            downloaded = info["function"](output_dir, args.num_images)

            print()
            print(f"✅ Downloaded {downloaded} images for {info['name']}")
            print("-" * 70)
            print()
    else:
        info = DATASET_TYPES[args.dataset_type]
        output_dir = args.output_dir or Path(info["output"])

        print(f"📦 Dataset: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"Use Case: {info['use_case']}")
        print(f"Output: {output_dir}")
        print()

        downloaded = info["function"](output_dir, args.num_images)

        print()
        print("=" * 70)
        print(f"✅ Downloaded {downloaded} images successfully!")
        print()
        print("Next steps:")
        print(f"1. Review images in {output_dir}/")
        print("2. Train LoRA:")
        print("   python -m ai_artist.training.train_lora \\")
        print(f"       --instance_data_dir {output_dir} \\")
        print(f"       --output_dir models/lora/{args.dataset_type}_style \\")
        print(f"       {info['training_params']}")
        print()
        print("3. Activate LoRA:")
        print(f"   python scripts/manage_loras.py set {args.dataset_type}_style")


if __name__ == "__main__":
    main()
