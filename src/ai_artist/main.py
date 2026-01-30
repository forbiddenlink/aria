"""Main application entry point."""

import asyncio
import io
import sys
from pathlib import Path

from PIL import Image

from .api.unsplash import UnsplashClient
from .core.controlnet import ControlNetPreprocessor
from .core.face_restore import FaceRestorer
from .core.generator import ImageGenerator
from .core.inpainter import ImageInpainter
from .core.upscaler import ImageUpscaler
from .curation.curator import ImageCurator
from .gallery.manager import GalleryManager
from .models.manager import ModelManager
from .scheduling.scheduler import CreationScheduler
from .trends.manager import TrendManager
from .utils.config import Config, get_torch_dtype, load_config
from .utils.logging import (
    PerformanceTimer,
    configure_logging,
    get_logger,
    set_request_id,
)
from .utils.prompt_engine import PromptEngine

logger = get_logger(__name__)


class AIArtist:
    """Main AI Artist application."""

    def __init__(self, config: Config):
        self.config = config
        self.generator = None
        self.upscaler = None
        self.inpainter = None
        self.face_restorer = None
        self.gallery = None
        self.unsplash = None
        self.scheduler = None
        self.curator = None
        self.prompt_engine = None
        self.trend_manager = None
        self.model_manager = None

        # Initialize all components
        self._initialize()

    def _initialize(self):
        """Initialize components."""
        # Setup logging with rotation
        configure_logging(
            log_level="INFO",
            log_file=Path("logs/ai_artist.log"),
            json_logs=False,  # Console-friendly for now
            enable_rotation=True,
        )

        logger.info("initializing_ai_artist")

        # Initialize prompt engine
        self.prompt_engine = PromptEngine()

        # Initialize generator
        self.generator = ImageGenerator(
            model_id=self.config.model.base_model,
            device=self.config.model.device,
            dtype=get_torch_dtype(self.config.model.dtype),
        )

        # Determine if ControlNet is enabled
        controlnet_model = None
        if self.config.controlnet.enabled:
            controlnet_model = self.config.controlnet.model_id
            logger.info("controlnet_enabled", model=controlnet_model)

        self.generator.load_model(controlnet_model=controlnet_model)

        # Load Refiner if enabled
        if self.config.model.use_refiner:
            logger.info("loading_refiner_enabled")
            self.generator.load_refiner(refiner_id=self.config.model.refiner_model)

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

        # Initialize upscaler if enabled
        if self.config.upscaling.enabled:
            self.upscaler = ImageUpscaler(
                model_id=self.config.upscaling.model_id,
                device=self.config.model.device,
                dtype=get_torch_dtype(self.config.model.dtype),
            )

        # Initialize inpainter if enabled
        if self.config.inpainting.enabled:
            self.inpainter = ImageInpainter(
                model_id=self.config.inpainting.model_id,
                device=self.config.model.device,
                dtype=get_torch_dtype(self.config.model.dtype),
            )

        # Initialize face restorer if enabled
        if self.config.face_restoration.enabled:
            self.face_restorer = FaceRestorer(
                model_path=self.config.face_restoration.model_path,
                device=self.config.model.device,
            )

        # Initialize gallery
        self.gallery = GalleryManager(Path("gallery"))

        # Initialize Unsplash
        self.unsplash = UnsplashClient(
            access_key=self.config.api_keys.unsplash_access_key.get_secret_value(),
        )

        # Initialize curator
        self.curator = ImageCurator(device=self.config.model.device)

        # Initialize scheduler
        self.scheduler = CreationScheduler()

        # Initialize trend manager
        if self.config.trends.enabled:
            self.trend_manager = TrendManager()

        # Initialize model manager
        if self.config.model_manager.enabled:
            self.model_manager = ModelManager(
                base_path=self.config.model_manager.base_path,
                api_key=self.config.model_manager.civitai_api_key,
            )

        logger.info("ai_artist_initialized")

    async def create_artwork(self, theme: str | None = None):
        """Create a single piece of artwork."""
        # Set unique request ID for this operation
        request_id = set_request_id()
        logger.info("creating_artwork", theme=theme, request_id=request_id)

        with PerformanceTimer(logger, "artwork_creation"):
            # Get inspiration from Unsplash
            query = theme or "art"
            if self.unsplash is None:
                raise RuntimeError("Unsplash client not initialized")
            photo = await self.unsplash.get_random_photo(query=query)

            # Build enhanced prompt with artistic styles using PromptEngine
            description = (
                photo.get("description") or photo.get("alt_description") or query
            )

            # Use dynamic prompt template
            # If description matches query exactly, it might mean no description was found
            if description.lower() == query.lower():
                # Fallback to a rich template
                template = (
                    "{masterpiece|best quality}, __styles__, __lighting__, __composition__, "
                    + query
                )
            else:
                template = f"{description}, __styles__, __lighting__, {{detailed|intricate|complex}}"

            if self.prompt_engine is None:
                raise RuntimeError("Prompt engine not initialized")

            # Autonomy Loop
            attempt = 0
            max_retries = (
                self.config.autonomy.max_retries if self.config.autonomy.enabled else 0
            )
            best_image = None
            best_score = -1.0

            while attempt <= max_retries:
                # Process prompt (fresh variation each time if retrying)
                prompt = self.prompt_engine.process(template)

                if attempt > 0:
                    logger.info(
                        "autonomy_retry",
                        attempt=attempt,
                        max_retries=max_retries,
                        new_prompt=prompt[:100],
                    )

                logger.info(
                    "got_inspiration",
                    query=query,
                    photo_id=photo["id"],
                    final_prompt=prompt,
                )

                # Prepare ControlNet image if enabled
                control_image = None
                if (
                    self.config.controlnet.enabled and attempt == 0
                ):  # Only load once usually, but keeping it simple
                    try:
                        logger.info(
                            "downloading_control_image", url=photo["urls"]["regular"]
                        )
                        if self.unsplash is None:
                            raise RuntimeError("Unsplash client not initialized")
                        image_data = await self.unsplash.download_image(
                            photo["urls"]["regular"]
                        )

                        source_image = Image.open(io.BytesIO(image_data))

                        with PerformanceTimer(logger, "controlnet_preprocessing"):
                            control_image = ControlNetPreprocessor.get_canny_image(
                                source_image,
                                low_threshold=self.config.controlnet.low_threshold,
                                high_threshold=self.config.controlnet.high_threshold,
                            )
                            # Resize to target generation size
                            control_image = control_image.resize(
                                (
                                    self.config.generation.width,
                                    self.config.generation.height,
                                )
                            )
                    except Exception as e:
                        logger.error("control_image_preparation_failed", error=str(e))

                # Generate images with performance tracking
                logger.info(
                    "generation_started",
                    num_variations=self.config.generation.num_variations,
                    prompt=prompt[:100],  # Truncate for logs
                )

                with PerformanceTimer(logger, "image_generation"):
                    if self.generator is None:
                        raise RuntimeError("Generator not initialized")
                    images = self.generator.generate(
                        prompt=prompt,
                        negative_prompt=self.config.generation.negative_prompt,
                        width=self.config.generation.width,
                        height=self.config.generation.height,
                        num_inference_steps=self.config.generation.num_inference_steps,
                        guidance_scale=self.config.generation.guidance_scale,
                        num_images=self.config.generation.num_variations,
                        use_refiner=self.config.model.use_refiner,
                        control_image=control_image,
                        controlnet_conditioning_scale=self.config.controlnet.conditioning_scale,
                    )

                # Evaluate and select best image
                logger.info("curation_started", num_images=len(images))

                current_batch_best_image = images[0]
                current_batch_best_score = 0.0
                scores = []

                with PerformanceTimer(logger, "image_curation"):
                    if self.curator is None:
                        raise RuntimeError("Curator not initialized")
                    for idx, image in enumerate(images, 1):
                        metrics = self.curator.evaluate(image, prompt)
                        score = metrics.overall_score
                        scores.append(score)
                        logger.debug(
                            "image_evaluated",
                            image_idx=idx,
                            score=round(score, 3),
                            aesthetic=round(metrics.aesthetic_score, 2),
                            clip=round(metrics.clip_score, 2),
                        )

                        if score > current_batch_best_score:
                            current_batch_best_score = score
                            current_batch_best_image = image

                logger.info(
                    "batch_best_selected",
                    best_score=round(current_batch_best_score, 3),
                    scores=[round(s, 3) for s in scores],
                )

                # Update global best if this batch is better
                if current_batch_best_score > best_score:
                    best_score = current_batch_best_score
                    best_image = current_batch_best_image

                # Check autonomy threshold
                if self.config.autonomy.enabled:
                    if best_score >= self.config.autonomy.min_score_threshold:
                        logger.info(
                            "autonomy_threshold_met",
                            score=best_score,
                            threshold=self.config.autonomy.min_score_threshold,
                        )
                        break
                    else:
                        logger.warning(
                            "autonomy_threshold_not_met",
                            score=best_score,
                            threshold=self.config.autonomy.min_score_threshold,
                        )
                        attempt += 1
                else:
                    break  # Not enabled, just run once

            # Fallback if we exhausted retries
            if best_image is None:
                raise RuntimeError("No valid image generated after all retries")

            # Upscale best image if enabled
            if self.config.upscaling.enabled and self.upscaler:
                with PerformanceTimer(logger, "image_upscaling"):
                    logger.info("upscaling_best_image")
                    try:
                        best_image = self.upscaler.upscale(
                            image=best_image,
                            prompt=prompt,
                            noise_level=self.config.upscaling.noise_level,
                        )
                    except Exception as e:
                        logger.error(
                            "upscaling_failed_outputting_original", error=str(e)
                        )

            # Apply face restoration if enabled
            if self.config.face_restoration.enabled and self.face_restorer:
                with PerformanceTimer(logger, "face_restoration"):
                    logger.info("restoring_faces")
                    try:
                        best_image = self.face_restorer.restore(best_image)
                    except Exception as e:
                        logger.error("face_restoration_failed", error=str(e))

            # Save best image
            if self.gallery is None:
                raise RuntimeError("Gallery not initialized")
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
        if self.unsplash is None:
            raise RuntimeError("Unsplash client not initialized")
        await self.unsplash.trigger_download(photo["links"]["download_location"])

        logger.info("artwork_created", path=str(saved_path))

        return saved_path

    async def update_trends(self):
        """Update trending styles."""
        if not self.trend_manager:
            return

        logger.info("updating_trends")
        try:
            await self.trend_manager.update_wildcard_file()
            if self.prompt_engine:
                self.prompt_engine.reload()

            # Auto-download models for new trends if enabled
            if (
                self.config.model_manager.enabled
                and self.config.model_manager.auto_download_trending
                and self.model_manager
            ):
                logger.info("checking_models_for_trends")
                trends = await self.trend_manager.get_combined_trends(
                    limit=5
                )  # Top 5 only
                for tag in trends:
                    # Async download in background essentially, or await if we want to ensure they are there
                    # For now await one by one
                    await self.model_manager.download_top_lora(tag)

            logger.info("trends_updated_successfully")
        except Exception as e:
            logger.error("trend_update_failed", error=str(e))

    async def run_manual(self, theme: str | None = None):
        """Run manual creation once."""
        # Optional: Update trends on manual run if enabled
        if self.config.trends.enabled:
            await self.update_trends()

        await self.create_artwork(theme=theme)

    async def run_automated(self):
        """Run in automated mode with scheduling."""
        logger.info("starting_automated_mode")

        # Initial trend update
        if self.config.trends.enabled:
            await self.update_trends()

        # Schedule daily creation at 9 AM
        def creation_job():
            asyncio.create_task(self.create_artwork())

        self.scheduler.schedule_daily(hour=9, minute=0, job_func=creation_job)

        # Schedule trend updates if enabled
        if self.config.trends.enabled:

            def trend_job():
                asyncio.create_task(self.update_trends())

            # Use interval scheduling for trends
            # Note: Scheduler wrapper might need update for interval, using daily for now
            # or just add another job.
            # Assuming schedule_daily is robust. Let's stick to update once a day before creation?
            # Or add a separate schedule.
            # self.scheduler.scheduler.add_job(trend_job, 'interval', hours=self.config.trends.update_interval_hours)
            # For simplicity, let's just do it at 8 AM
            self.scheduler.schedule_daily(hour=8, minute=0, job_func=trend_job)

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
        sys.stderr.write(f"Error: Config file not found at {config_path}\n")
        sys.stderr.write(
            "Please create config/config.yaml from config/config.example.yaml\n"
        )
        sys.exit(1)
    except Exception as e:
        logger.error("config_load_error", error=str(e))
        sys.stderr.write(f"Error loading config: {e}\n")
        sys.exit(1)

    app = AIArtist(config)

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
