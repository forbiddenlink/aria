#!/usr/bin/env python3
"""Cleanup script to remove black/blank/invalid images from the gallery.

USAGE:
    # Dry run - see what would be deleted without actually deleting
    python scripts/cleanup_gallery.py --dry-run

    # Actually delete invalid images
    python scripts/cleanup_gallery.py

    # Specify custom gallery path
    python scripts/cleanup_gallery.py --gallery-path /path/to/gallery
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_artist.gallery.manager import GalleryManager
from ai_artist.utils.logging import configure_logging, get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Remove black/blank/invalid images from the gallery"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--gallery-path",
        type=Path,
        default=Path("gallery"),
        help="Path to gallery directory (default: gallery/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    configure_logging(
        log_level="DEBUG" if args.verbose else "INFO",
        json_logs=False,
    )
    logger = get_logger(__name__)

    # Verify gallery exists
    if not args.gallery_path.exists():
        logger.error("gallery_not_found", path=str(args.gallery_path))
        print(f"Error: Gallery path does not exist: {args.gallery_path}")
        sys.exit(1)

    # Initialize gallery manager
    gallery = GalleryManager(args.gallery_path)

    # Run cleanup
    print(f"\nScanning gallery: {args.gallery_path}")
    if args.dry_run:
        print("DRY RUN - no files will be deleted\n")
    else:
        print("WARNING: This will permanently delete invalid images!\n")

    stats = gallery.cleanup_invalid_images(dry_run=args.dry_run)

    # Print results
    print("\n" + "=" * 50)
    print("CLEANUP RESULTS")
    print("=" * 50)
    print(f"Images scanned:  {stats['scanned']}")
    print(f"Images deleted:  {stats['deleted']}")
    print(f"Errors:          {stats['errors']}")

    if stats["deleted_files"]:
        print(f"\n{'Would delete' if args.dry_run else 'Deleted'} files:")
        for f in stats["deleted_files"][:20]:  # Show first 20
            print(f"  - {f}")
        if len(stats["deleted_files"]) > 20:
            print(f"  ... and {len(stats['deleted_files']) - 20} more")

    if stats["error_files"]:
        print("\nFiles with errors:")
        for f in stats["error_files"][:10]:
            print(f"  - {f}")

    if args.dry_run and stats["deleted"] > 0:
        print(f"\nRun without --dry-run to delete {stats['deleted']} invalid images")


if __name__ == "__main__":
    main()
