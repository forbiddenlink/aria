#!/usr/bin/env python3
"""CLI tool to launch the web gallery."""

import sys

import uvicorn

from ai_artist.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def main():
    """Launch the web gallery server."""
    logger.info("starting_web_gallery", host="0.0.0.0", port=8000)

    print("\n🎨 AI Artist Web Gallery")
    print("=" * 50)
    print("\n📍 Gallery URL: http://localhost:8000")
    print("📊 API Docs:    http://localhost:8000/docs")
    print("❤️  Health:     http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    print("=" * 50)

    try:
        uvicorn.run(
            "ai_artist.web.app:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False,
            ws="websockets",  # Explicitly force websockets
        )
    except KeyboardInterrupt:
        logger.info("web_gallery_stopped")
        print("\n\n👋 Gallery server stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
