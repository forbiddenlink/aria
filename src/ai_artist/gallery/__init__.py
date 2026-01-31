"""Gallery management for AI Artist.

This module provides the GalleryManager for organizing and viewing generated images.
"""

from .manager import GalleryManager
from .viewer import main as viewer_main

__all__ = ["GalleryManager", "viewer_main"]
