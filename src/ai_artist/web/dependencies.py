"""FastAPI dependency injection functions for shared resources."""

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, HTTPException

from ..gallery.manager import GalleryManager
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Global instances (initialized in lifespan)
_gallery_manager: GalleryManager | None = None
_gallery_path: str | None = None


def set_gallery_manager(manager: GalleryManager, path: str) -> None:
    """Set global gallery manager instance (called during app startup)."""
    global _gallery_manager, _gallery_path
    _gallery_manager = manager
    _gallery_path = path


def get_gallery_manager() -> Generator[GalleryManager, None, None]:
    """Dependency that provides gallery manager instance."""
    if _gallery_manager is None:
        raise HTTPException(status_code=503, detail="Gallery not initialized")
    yield _gallery_manager


def get_gallery_path() -> str:
    """Dependency that provides gallery path."""
    if _gallery_path is None:
        raise HTTPException(status_code=503, detail="Gallery not initialized")
    return _gallery_path


# Type annotations for cleaner dependency injection
GalleryManagerDep = Annotated[GalleryManager, Depends(get_gallery_manager)]
GalleryPathDep = Annotated[str, Depends(get_gallery_path)]
