"""Gallery management for storing generated images."""

import json
from datetime import datetime
from pathlib import Path

from PIL import Image, PngImagePlugin

from ..curation.curator import is_black_or_blank
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GalleryManager:
    """Manage gallery storage and organization."""

    def __init__(self, gallery_path: Path):
        self.gallery_path = gallery_path
        self.gallery_path.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        image: Image.Image,
        prompt: str,
        metadata: dict,
        featured: bool = False,
    ) -> Path:
        """Save image with metadata."""
        # Create directory structure: year/month/day
        now = datetime.now()
        save_dir = (
            self.gallery_path
            / f"{now.year}"
            / f"{now.month:02d}"
            / f"{now.day:02d}"
        )

        if featured:
            save_dir = save_dir / "featured"
        else:
            save_dir = save_dir / "archive"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{metadata.get('seed', 'noseed')}.png"
        image_path = save_dir / filename

        # Add metadata to PNG
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("prompt", prompt)
        pnginfo.add_text("metadata", json.dumps(metadata))
        pnginfo.add_text("AI-Generated", "true")  # EU AI Act compliance

        # Save image
        image.save(image_path, pnginfo=pnginfo)

        # Save sidecar metadata file
        metadata_path = image_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "metadata": metadata,
                    "created_at": now.isoformat(),
                    "featured": featured,
                },
                indent=2,
            )
        )

        logger.info("image_saved", path=str(image_path), featured=featured)

        return image_path

    def list_images(self, featured_only: bool = False) -> list[Path]:
        """List all images in gallery."""
        pattern = "**/featured/*.png" if featured_only else "**/*.png"
        return sorted(self.gallery_path.glob(pattern), reverse=True)

    def cleanup_invalid_images(self, dry_run: bool = False) -> dict[str, int | list[str]]:
        """Scan gallery and remove black/blank/invalid images.

        Args:
            dry_run: If True, only report what would be deleted without deleting

        Returns:
            Dict with cleanup statistics
        """
        # Track statistics with proper types
        scanned = 0
        deleted = 0
        errors = 0
        deleted_files: list[str] = []
        error_files: list[str] = []

        all_images = self.list_images()
        logger.info("cleanup_starting", total_images=len(all_images), dry_run=dry_run)

        for img_path in all_images:
            scanned += 1

            try:
                # Load and check image
                with Image.open(img_path) as img:
                    is_invalid, reason = is_black_or_blank(img)

                if is_invalid:
                    logger.info(
                        "cleanup_found_invalid",
                        path=str(img_path),
                        reason=reason,
                        dry_run=dry_run,
                    )

                    if not dry_run:
                        # Delete image file
                        img_path.unlink()

                        # Delete associated metadata JSON
                        metadata_path = img_path.with_suffix(".json")
                        if metadata_path.exists():
                            metadata_path.unlink()

                    deleted += 1
                    deleted_files.append(str(img_path))

            except Exception as e:
                logger.error("cleanup_error", path=str(img_path), error=str(e))
                errors += 1
                error_files.append(str(img_path))

        logger.info(
            "cleanup_complete",
            scanned=scanned,
            deleted=deleted,
            errors=errors,
            dry_run=dry_run,
        )

        return {
            "scanned": scanned,
            "deleted": deleted,
            "errors": errors,
            "deleted_files": deleted_files,
            "error_files": error_files,
        }

