"""FastAPI web gallery application."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

from ..gallery.manager import GalleryManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ImageMetadata(BaseModel):
    """Image metadata response model."""

    path: str
    filename: str
    prompt: str
    created_at: str
    featured: bool
    metadata: dict
    thumbnail_url: str
    full_url: str


class GalleryStats(BaseModel):
    """Gallery statistics response model."""

    total_images: int
    featured_images: int
    total_prompts: int
    date_range: dict


# Initialize FastAPI app
app = FastAPI(
    title="AI Artist Gallery",
    description="Browse and explore AI-generated artwork",
    version="1.0.0",
)

# Gallery manager (will be initialized on startup)
gallery_manager: GalleryManager | None = None
gallery_path: Path | None = None

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@app.on_event("startup")
async def startup_event():
    """Initialize gallery on startup."""
    global gallery_manager, gallery_path
    gallery_path = Path("gallery")
    gallery_manager = GalleryManager(gallery_path)
    logger.info("web_gallery_started", gallery_path=str(gallery_path))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve gallery homepage."""
    return templates.TemplateResponse(
        "gallery.html",
        {"request": request, "title": "AI Artist Gallery"},
    )


@app.get("/api/images", response_model=list[ImageMetadata])
async def list_images(
    featured: bool | None = Query(None, description="Filter by featured status"),
    limit: int = Query(50, ge=1, le=500, description="Number of images to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    search: str | None = Query(None, description="Search in prompts"),
):
    """List all images with metadata."""
    if not gallery_manager:
        raise HTTPException(status_code=503, detail="Gallery not initialized")

    # Get image paths
    image_paths = gallery_manager.list_images(featured_only=bool(featured))

    # Filter by search term
    if search:
        filtered_paths = []
        for img_path in image_paths:
            metadata_path = img_path.with_suffix(".json")
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                    if search.lower() in metadata.get("prompt", "").lower():
                        filtered_paths.append(img_path)
                except Exception:
                    continue
        image_paths = filtered_paths

    # Apply pagination
    total = len(image_paths)
    image_paths = image_paths[offset : offset + limit]

    # Build response
    results = []
    for img_path in image_paths:
        metadata_path = img_path.with_suffix(".json")
        if not metadata_path.exists():
            continue

        try:
            metadata = json.loads(metadata_path.read_text())
            if gallery_path is None:
                continue
            relative_path = img_path.relative_to(gallery_path)

            # Skip images without prompts (blank/incomplete metadata)
            prompt = metadata.get("prompt", "")
            if not prompt or not prompt.strip():
                logger.debug("skipping_image_without_prompt", path=str(img_path))
                continue

            # Skip black or corrupted images
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img.convert("RGB"))
                    # Check if image is mostly black (mean brightness < 10)
                    if img_array.mean() < 10:
                        logger.debug("skipping_black_image", path=str(img_path))
                        continue
            except Exception as e:
                logger.debug(
                    "skipping_corrupted_image", path=str(img_path), error=str(e)
                )
                continue

            results.append(
                ImageMetadata(
                    path=str(relative_path),
                    filename=img_path.name,
                    prompt=prompt,
                    created_at=metadata.get("created_at", ""),
                    featured=metadata.get("featured", False),
                    metadata=metadata.get("metadata", {}),
                    thumbnail_url=f"/api/images/file/{relative_path}",
                    full_url=f"/api/images/file/{relative_path}",
                )
            )
        except Exception as e:
            logger.warning("failed_to_load_metadata", path=str(img_path), error=str(e))
            continue

    logger.info(
        "images_listed",
        total=total,
        returned=len(results),
        featured=featured,
        search=search,
    )

    return results


@app.get("/api/images/file/{file_path:path}")
async def get_image_file(file_path: str):
    """Serve image file."""
    if not gallery_path:
        raise HTTPException(status_code=503, detail="Gallery not initialized")

    full_path = gallery_path / file_path

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    # Security: ensure path is within gallery
    try:
        full_path.resolve().relative_to(gallery_path.resolve())
    except ValueError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e

    return FileResponse(full_path, media_type="image/png")


@app.get("/api/stats", response_model=GalleryStats)
async def get_stats():
    """Get gallery statistics."""
    if not gallery_manager:
        raise HTTPException(status_code=503, detail="Gallery not initialized")

    all_images = gallery_manager.list_images(featured_only=False)
    featured_images = gallery_manager.list_images(featured_only=True)

    # Collect unique prompts and date range
    prompts = set()
    dates = []

    for img_path in all_images:
        metadata_path = img_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                prompts.add(metadata.get("prompt", ""))
                created_at = metadata.get("created_at", "")
                if created_at:
                    dates.append(created_at)
            except Exception as e:
                logger.debug("failed_to_parse_metadata", error=str(e))
                continue

    date_range = {}
    if dates:
        dates.sort()
        date_range = {"earliest": dates[0], "latest": dates[-1]}

    return GalleryStats(
        total_images=len(all_images),
        featured_images=len(featured_images),
        total_prompts=len(prompts),
        date_range=date_range,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gallery_initialized": gallery_manager is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
