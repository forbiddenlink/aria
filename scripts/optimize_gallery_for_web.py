#!/usr/bin/env python3
"""Optimize gallery images for web deployment.

This script:
1. Converts images to WebP format for better compression
2. Creates thumbnails for faster loading
3. Generates a manifest JSON for the gallery
4. Optimizes file sizes while maintaining quality
"""

import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def optimize_image_for_web(
    image_path: Path,
    output_dir: Path,
    max_size: int = 1200,
    thumbnail_size: int = 400,
    quality: int = 85,
) -> dict[str, str]:
    """Optimize a single image for web deployment.

    Args:
        image_path: Path to the original image
        output_dir: Directory to save optimized images
        max_size: Maximum dimension for full-size image
        thumbnail_size: Maximum dimension for thumbnail
        quality: WebP quality (0-100)

    Returns:
        Dictionary with paths to optimized images
    """
    try:
        img = Image.open(image_path)

        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            # Create a white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Create relative path structure
        rel_path = image_path.relative_to(Path("gallery"))
        full_output_path = output_dir / rel_path.parent / f"{rel_path.stem}.webp"
        thumb_output_path = output_dir / rel_path.parent / f"{rel_path.stem}_thumb.webp"

        # Create output directories
        full_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Optimize full-size image
        full_img = img.copy()
        if max(full_img.size) > max_size:
            full_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        full_img.save(full_output_path, "WEBP", quality=quality, method=6)

        # Create thumbnail
        thumb_img = img.copy()
        thumb_img.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
        thumb_img.save(thumb_output_path, "WEBP", quality=75, method=6)

        return {
            "original": str(image_path),
            "optimized": str(full_output_path),
            "thumbnail": str(thumb_output_path),
            "original_size": image_path.stat().st_size,
            "optimized_size": full_output_path.stat().st_size,
            "thumbnail_size": thumb_output_path.stat().st_size,
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def generate_gallery_manifest(gallery_dir: Path, output_file: Path) -> dict:
    """Generate a JSON manifest of all gallery images.

    Args:
        gallery_dir: Path to the gallery directory
        output_file: Path to save the manifest JSON

    Returns:
        Manifest dictionary
    """
    manifest = {
        "images": [],
        "stats": {
            "total_images": 0,
            "total_size_original": 0,
            "total_size_optimized": 0,
            "compression_ratio": 0,
        },
    }

    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(gallery_dir.rglob(f"*{ext}"))

    # Filter out thumbnails and already optimized files
    image_files = [
        f
        for f in image_files
        if "_thumb" not in f.stem and f.parent.name != "optimized"
    ]

    manifest["stats"]["total_images"] = len(image_files)

    for img_path in image_files:
        try:
            # Get relative path
            rel_path = img_path.relative_to(gallery_dir)

            # Load metadata if available
            metadata_path = img_path.with_suffix(".json")
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

            manifest["images"].append(
                {
                    "path": str(rel_path),
                    "filename": img_path.name,
                    "size": img_path.stat().st_size,
                    "metadata": metadata,
                }
            )
        except Exception as e:
            print(f"Error processing metadata for {img_path}: {e}")

    # Save manifest
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    """Main optimization script."""
    print("🎨 AI Artist - Gallery Optimization for Web Deployment")
    print("=" * 60)

    gallery_dir = Path("gallery")
    output_dir = Path("gallery_optimized")

    if not gallery_dir.exists():
        print("❌ Gallery directory not found!")
        return

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(gallery_dir.rglob(f"*{ext}"))

    # Filter out thumbnails
    image_files = [f for f in image_files if "_thumb" not in f.stem]

    print(f"\n📊 Found {len(image_files)} images to optimize")

    if not image_files:
        print("No images to optimize!")
        return

    # Optimize images
    results = []
    total_original = 0
    total_optimized = 0

    print("\n🔄 Optimizing images...")
    for img_path in tqdm(image_files, desc="Processing"):
        result = optimize_image_for_web(img_path, output_dir)
        if result:
            results.append(result)
            total_original += result["original_size"]
            total_optimized += result["optimized_size"]

    # Generate manifest
    print("\n📝 Generating gallery manifest...")
    generate_gallery_manifest(gallery_dir, output_dir / "manifest.json")

    # Display results
    print("\n✅ Optimization Complete!")
    print("📊 Results:")
    print(f"   - Images processed: {len(results)}")
    print(f"   - Original size: {total_original / 1e6:.2f} MB")
    print(f"   - Optimized size: {total_optimized / 1e6:.2f} MB")
    print(
        f"   - Space saved: {(total_original - total_optimized) / 1e6:.2f} MB ({(1 - total_optimized / total_original) * 100:.1f}%)"
    )
    print(f"\n📁 Optimized gallery saved to: {output_dir}")
    print(f"📄 Manifest saved to: {output_dir / 'manifest.json'}")
    print("\n💡 Next steps:")
    print(f"   1. Review optimized images in {output_dir}")
    print("   2. Run: vercel --prod")
    print("   3. Your gallery will be live!")


if __name__ == "__main__":
    main()
