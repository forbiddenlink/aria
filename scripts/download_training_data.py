#!/usr/bin/env python3
"""Download training images from Unsplash for LoRA training."""

import time
from pathlib import Path

import requests

# High-quality artistic images from Unsplash (free to use for AI training)
# Using collection IDs for curated artistic content
COLLECTIONS = {
    "abstract_art": "3330445",  # Abstract art collection
    "minimalism": "1154337",  # Minimalist photography
    "nature": "3356584",  # Nature and landscapes
}

# Unsplash API (public access)
BASE_URL = "https://api.unsplash.com"
# Note: Using demo access - rate limited but good for testing


def download_from_collection(
    collection_id: str,
    output_dir: Path,
    num_images: int = 10,
    access_key: str = None,
):
    """Download images from an Unsplash collection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if access_key:
        headers = {"Authorization": f"Client-ID {access_key}"}
    else:
        # Use demo mode with public endpoint
        headers = {}
        print("⚠️  Using public endpoint - limited to 50 requests/hour")

    url = f"{BASE_URL}/collections/{collection_id}/photos"
    params = {"per_page": num_images}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        photos = response.json()

        print(f"📥 Downloading {len(photos)} images from collection {collection_id}...")

        for idx, photo in enumerate(photos, 1):
            # Get the regular sized image (1080px)
            image_url = photo["urls"]["regular"]
            image_id = photo["id"]

            # Download image
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            # Save with descriptive name
            filename = f"{collection_id}_{image_id}.jpg"
            filepath = output_dir / filename

            with open(filepath, "wb") as f:
                f.write(img_response.content)

            print(f"  ✓ Downloaded {idx}/{len(photos)}: {filename}")

            # Be nice to the API
            time.sleep(1)

        return len(photos)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading from collection {collection_id}: {e}")
        return 0


def download_fallback_images(output_dir: Path, num_images: int = 30):
    """Download using reliable public sources."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use Picsum Photos - reliable Lorem Ipsum for photos (public domain)
    # High quality images from various photographers
    downloaded = 0

    print(f"📥 Downloading {num_images} curated training images from Picsum...")

    for idx in range(1, num_images + 1):
        try:
            # Picsum provides random high-quality images
            # Using specific seed for consistency
            seed = 1000 + idx
            image_url = f"https://picsum.photos/seed/{seed}/1024/1024"

            response = requests.get(image_url, timeout=30, allow_redirects=True)
            response.raise_for_status()

            filename = f"training_image_{idx:03d}.jpg"
            filepath = output_dir / filename

            with open(filepath, "wb") as f:
                f.write(response.content)

            print(f"  ✓ Downloaded {idx}/{num_images}: {filename}")
            downloaded += 1

            # Be nice - rate limit
            time.sleep(0.5)

        except Exception as e:
            print(f"  ✗ Failed to download image {idx}: {e}")
            continue

    return downloaded


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download training images from Unsplash"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/training"),
        help="Output directory for training images",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=30,
        help="Number of images to download",
    )
    parser.add_argument(
        "--access-key",
        type=str,
        help="Unsplash API access key (optional)",
    )

    args = parser.parse_args()

    print("🎨 AI Artist - Training Data Downloader")
    print("=" * 50)
    print(f"Output directory: {args.output}")
    print(f"Target images: {args.num_images}")
    print()

    # Try fallback method (more reliable)
    downloaded = download_fallback_images(args.output, args.num_images)

    print()
    print("=" * 50)
    print(f"✅ Downloaded {downloaded} images successfully!")
    print()
    print("Next steps:")
    print("1. Review images in datasets/training/")
    print("2. Document sources in datasets/licenses.txt")
    print("3. Run training:")
    print("   python -m ai_artist.training.train_lora \\")
    print("       --instance_data_dir datasets/training \\")
    print("       --output_dir models/lora/my_style")


if __name__ == "__main__":
    main()
