"""Admin dashboard for AI Artist."""

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..db.models import Artwork
from ..db.session import get_db_session
from ..utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# Setup templates
templates_dir = Path(__file__).parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request) -> HTMLResponse:
    """Render admin dashboard.

    Args:
        request: FastAPI request

    Returns:
        HTML response with dashboard
    """
    return templates.TemplateResponse(
        "admin/dashboard.html",
        {"request": request, "title": "Admin Dashboard"},
    )


@router.get("/stats")
async def get_statistics(
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """Get system statistics.

    Args:
        db: Database session

    Returns:
        Dict with statistics
    """
    try:
        # Total artworks
        total_artworks = db.query(Artwork).count()

        # Artworks by mood
        from sqlalchemy import func

        moods = (
            db.query(Artwork.mood, func.count(Artwork.id)).group_by(Artwork.mood).all()
        )

        # Recent artworks
        recent = db.query(Artwork).order_by(Artwork.created_at.desc()).limit(10).all()

        # Top rated
        top_rated = db.query(Artwork).order_by(Artwork.score.desc()).limit(10).all()

        return {
            "total_artworks": total_artworks,
            "moods": dict(moods),
            "recent": [
                {
                    "id": a.id,
                    "theme": a.theme,
                    "mood": a.mood,
                    "score": a.score,
                    "created_at": a.created_at.isoformat(),
                }
                for a in recent
            ],
            "top_rated": [
                {
                    "id": a.id,
                    "theme": a.theme,
                    "score": a.score,
                    "image_path": a.image_path,
                }
                for a in top_rated
            ],
        }

    except Exception as e:
        logger.error("stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/performance")
async def get_performance_metrics() -> dict[str, Any]:
    """Get performance metrics.

    Returns:
        Dict with performance data
    """
    try:
        from ..monitoring.metrics import GENERATION_REQUESTS_TOTAL

        return {
            "total_generations": (
                GENERATION_REQUESTS_TOTAL._value.get()
                if hasattr(GENERATION_REQUESTS_TOTAL, "_value")
                else 0
            ),
            "avg_duration": 0,  # TODO: Calculate from histogram
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("performance_error", error=str(e))
        return {
            "total_generations": 0,
            "avg_duration": 0,
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/system")
async def get_system_info() -> dict[str, Any]:
    """Get system information.

    Returns:
        Dict with system info
    """
    try:
        import psutil
        import torch

        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "gpu": {
                "available": torch.cuda.is_available(),
                "device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
                "device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "memory_allocated": (
                    torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
                ),
                "memory_reserved": (
                    torch.cuda.memory_reserved(0) if torch.cuda.is_available() else 0
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("system_info_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/artworks/{artwork_id}")
async def delete_artwork(
    artwork_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """Delete artwork.

    Args:
        artwork_id: Artwork ID
        db: Database session

    Returns:
        Success message
    """
    try:
        artwork = db.query(Artwork).filter(Artwork.id == artwork_id).first()

        if not artwork:
            raise HTTPException(status_code=404, detail="Artwork not found")

        # Delete image file if exists
        if artwork.image_path:
            image_path = Path(artwork.image_path)
            if image_path.exists():
                image_path.unlink()

        db.delete(artwork)
        db.commit()

        logger.info("artwork_deleted", artwork_id=artwork_id)

        return {"message": "Artwork deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_error", artwork_id=artwork_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Clear application cache.

    Returns:
        Success message
    """
    try:
        from ..caching import RedisCache

        cache = RedisCache()
        cleared = await cache.clear_pattern("aria:*")

        logger.info("cache_cleared", keys_deleted=cleared)

        return {
            "message": f"Cache cleared: {cleared} keys deleted",
        }

    except Exception as e:
        logger.error("cache_clear_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
