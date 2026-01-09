"""Main application entry point."""

import asyncio
import sys
from pathlib import Path

from .api.unsplash import UnsplashClient
from .core.generator import ImageGenerator
from .curation.curator import ImageCurator
from .gallery.manager import GalleryManager
from .scheduling.scheduler import CreationScheduler
from .utils.config import Config, get_torch_dtype, load_config
from .utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


class AIArtist:
    """Main AI Artist application."""

    def __init__(self, config: Config):
        self.config = config
        self.generator = None
        self.gallery = None
        self.unsplash = None
        self.scheduler = None
        self.curator = None

    def initialize(self):
        """Initialize components."""
        logger.info("initializing_ai_artist")

        # Setup logging
        configure_logging(log_level="INFO", log_file=Path("logs/ai_artist.log"))

        # Initialize generator
        self.generator = ImageGenerator(
            model_id=self.config.model.base_model,
            device=self.config.model.device,
            dtype=get_torch_dtype(self.config.model.dtype),
        )
        self.generator.load_model()

        # Load LoRA if specified
        if self.config.model.lora_path:
            lora_path = Path(self.config.model.lora_path)
            if lora_path.exists():
                logger.info("loading_lora_from_config", path=str(lora_path))
                self.generator.load_lora(
                    lora_path=lora_path,
                    lora_scale=self.config.model.lora_scale,
                )
            else:
                logger.warning("lora_path_not_found", path=str(lora_path))

        # Initialize gallery
        self.gallery = GalleryManager(Path("gallery"))

        # Initialize Unsplash
        self.unsplash = UnsplashClient(
            access_key=self.config.api_keys.unsplash_access_key,
        )

        # Initialize curator
        self.curator = ImageCurator(device=self.config.model.device)

        # Initialize scheduler
        self.scheduler = CreationScheduler()

        logger.info("ai_artist_initialized")

    async def create_artwork(self, theme: str | None = None):
        """Create a single piece of artwork."""
        logger.info("creating_artwork", theme=theme)

        # Get inspiration from Unsplash
        query = theme or "art"
        photo = await self.unsplash.get_random_photo(query=query)

        # Build enhanced prompt with artistic styles
        description = photo.get("description") or photo.get("alt_description") or query
        import random

        styles = [
            "masterpiece, highly detailed, professional photography",
            "artistic interpretation, vivid colors, detailed composition",
            "beautiful artwork, intricate details, stunning visual",
            "creative composition, high quality, aesthetically pleasing",
        ]
        style_modifier = random.choice(styles)
        prompt = f"{description}, {style_modifier}"

        logger.info("got_inspiration", query=query, photo_id=photo["id"])

        # Generate images with progress callback
        print(f"\n🎨 Generating {self.config.generation.num_variations} variations...")
        images = self.generator.generate(
            prompt=prompt,
            negative_prompt=self.config.generation.negative_prompt,
            width=self.config.generation.width,
            height=self.config.generation.height,
            num_inference_steps=self.config.generation.num_inference_steps,
            guidance_scale=self.config.generation.guidance_scale,
            num_images=self.config.generation.num_variations,
        )

        # Evaluate and select best image
        print("\n🔍 Evaluating image quality...")
        best_image = images[0]
        best_score = 0.0

        for idx, image in enumerate(images, 1):
            metrics = self.curator.evaluate(image, prompt)
            score = metrics.overall_score
            print(
                f"   Image {idx}: score={score:.3f} (aesthetic={metrics.aesthetic_score:.2f}, clip={metrics.clip_score:.2f})"
            )

            if score > best_score:
                best_score = score
                best_image = image

        print(f"\n✨ Selected best image with score: {best_score:.3f}")

        # Save best image
        saved_path = self.gallery.save_image(
            image=best_image,
            prompt=prompt,
            metadata={
                "source_url": photo["urls"]["regular"],
                "source_id": photo["id"],
                "theme": theme,
                "model": self.config.model.base_model,
                "quality_score": float(best_score),
            },
        )

        # Track download
        await self.unsplash.trigger_download(photo["links"]["download_location"])

        logger.info("artwork_created", path=str(saved_path))

        return saved_path

    async def run_manual(self, theme: str | None = None):
        """Run manual creation once."""
        await self.create_artwork(theme=theme)

    async def run_automated(self):
        """Run in automated mode with scheduling."""
        logger.info("starting_automated_mode")

        # Schedule daily creation at 9 AM
        def creation_job():
            asyncio.create_task(self.create_artwork())

        self.scheduler.schedule_daily(hour=9, minute=0, job_func=creation_job)

        logger.info("automated_mode_running")

        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("interrupted_by_user")

    async def shutdown(self):
        """Cleanup resources."""
        logger.info("shutting_down")

        if self.generator:
            self.generator.unload()

        if self.unsplash:
            await self.unsplash.close()

        if self.scheduler:
            self.scheduler.shutdown()

        logger.info("shutdown_complete")


async def async_main(config_path: Path, mode: str = "manual", theme: str | None = None):
    """Async main function."""
    # Load config
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error("config_not_found", path=str(config_path))
        print(f"Error: Config file not found at {config_path}")
        print("Please create config/config.yaml from config/config.example.yaml")
        sys.exit(1)
    except Exception as e:
        logger.error("config_load_error", error=str(e))
        print(f"Error loading config: {e}")
        sys.exit(1)

    app = AIArtist(config)
    app.initialize()

    try:
        if mode == "manual":
            await app.run_manual(theme=theme)
        else:
            await app.run_automated()
    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
    except Exception as e:
        logger.error("application_error", error=str(e), exc_info=True)
        raise
    finally:
        await app.shutdown()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AI Artist - Autonomous Art Generator")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "auto"],
        default="manual",
        help="Run mode: manual (one-time) or auto (scheduled)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Theme for artwork (e.g., 'sunset', 'mountains')",
    )
    args = parser.parse_args()

    # Run async main
    asyncio.run(async_main(args.config, args.mode, args.theme))


if __name__ == "__main__":
    main()
