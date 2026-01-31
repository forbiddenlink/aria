"""Vercel serverless function entry point for AI Artist Gallery.

This provides a gallery-only mode for Vercel deployment.
For full image generation with GPU support, use Docker deployment
on Railway, Render, or a GPU-enabled cloud provider.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Create a lightweight app for Vercel
app = FastAPI(
    title="AI Artist Gallery (Vercel)",
    description="Browse AI-generated artwork - Gallery only mode",
    version="1.0.0",
)

# Add CORS
# Configure CORS - restrict in production!
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://aria-gallery.vercel.app",  # Update with your actual domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Gallery homepage."""
    return {
        "name": "AI Artist Gallery",
        "mode": "gallery-only",
        "message": "This is a gallery-only deployment. For image generation, deploy with Docker.",
        "endpoints": {
            "images": "/api/images",
            "stats": "/api/stats",
            "health": "/health",
        },
    }


@app.get("/api/images")
async def vercel_list_images(
    featured: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
):
    """List gallery images (read-only in Vercel mode)."""
    # In Vercel mode, we serve from a static gallery or S3
    gallery_path = os.environ.get("GALLERY_PATH", "/tmp/gallery")

    if not Path(gallery_path).exists():
        return JSONResponse(
            content={
                "message": "Gallery not configured for Vercel deployment",
                "hint": "Set GALLERY_PATH environment variable or use external storage",
            },
            status_code=503,
        )

    # Return empty for now - in production, connect to S3/external storage
    return []


@app.get("/api/stats")
async def vercel_stats():
    """Gallery statistics."""
    return {
        "total_images": 0,
        "featured_images": 0,
        "total_prompts": 0,
        "date_range": {},
        "mode": "gallery-only",
    }


@app.post("/api/generate")
async def vercel_generate():
    """Generation not available in Vercel mode."""
    raise HTTPException(
        status_code=503,
        detail={
            "error": "Image generation is not available in gallery-only mode",
            "reason": "Vercel serverless functions don't support GPU workloads",
            "alternatives": [
                "Deploy with Docker on Railway (https://railway.app)",
                "Deploy with Docker on Render (https://render.com)",
                "Self-host with Docker on a GPU server",
            ],
        },
    )


@app.get("/health")
async def vercel_health():
    """Health check."""
    return {
        "status": "healthy",
        "mode": "gallery-only",
        "platform": "vercel",
        "generation_available": False,
    }


# Export for Vercel
handler = app
