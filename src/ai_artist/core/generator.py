"""Image generation using Stable Diffusion + LoRA."""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image

from ..curation.curator import is_black_or_blank
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ImageGenerator:
    """Stable Diffusion image generator with LoRA support.

    Can be used as a context manager for automatic cleanup:
        with ImageGenerator() as generator:
            generator.load_model()
            images = generator.generate("a beautiful landscape")

    Supports multi-model caching for mood-based model selection.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Literal["cuda", "mps", "cpu"] = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        self.refiner = None

        # Cache for loaded models to avoid reloading
        self._model_cache: dict[str, DiffusionPipeline] = {}
        self._current_model_id: str | None = None

        # MPS + float16 can produce black/NaN images - warn and suggest float32
        if device == "mps" and dtype == torch.float16:
            logger.warning(
                "mps_float16_instability",
                message="MPS with float16 may produce black images. "
                "Consider using float32 for more stable results.",
                hint="Set dtype: 'float32' in config or use DEVICE=cuda if available",
            )

        logger.info(
            "initializing_generator", model=model_id, device=device, dtype=str(dtype)
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup resources."""
        self.unload()
        return False

    def get_model_for_mood(
        self, mood: str, mood_models: dict[str, str] | None = None
    ) -> str:
        """Get the appropriate model ID for a given mood.

        Args:
            mood: The current mood (e.g., "contemplative", "chaotic")
            mood_models: Optional mood-to-model mapping dict

        Returns:
            The model ID to use for this mood
        """
        if mood_models is None:
            return self.model_id

        model_id = mood_models.get(mood.lower(), self.model_id)
        logger.debug("model_for_mood", mood=mood, model=model_id)
        return model_id

    def switch_model(
        self, new_model_id: str, controlnet_model: str | None = None
    ) -> bool:
        """Switch to a different model, using cache if available.

        Args:
            new_model_id: The model ID to switch to
            controlnet_model: Optional ControlNet model

        Returns:
            True if model was switched, False if already using this model
        """
        if new_model_id == self._current_model_id and self.pipeline is not None:
            logger.debug("model_already_loaded", model=new_model_id)
            return False

        # Check cache first
        if new_model_id in self._model_cache:
            logger.info("switching_to_cached_model", model=new_model_id)
            self.pipeline = self._model_cache[new_model_id]
            self._current_model_id = new_model_id
            return True

        # Load new model (will be cached)
        old_model_id = self.model_id
        self.model_id = new_model_id
        self.load_model(controlnet_model=controlnet_model)
        self.model_id = old_model_id  # Restore original default

        return True

    def load_model(
        self, controlnet_model: str | None = None, model_override: str | None = None
    ):
        """Load the diffusion pipeline.

        Args:
            controlnet_model: Optional ControlNet model to load
            model_override: Optional model ID to load instead of self.model_id
        """
        model_to_load = model_override or self.model_id
        logger.info("loading_model", model=model_to_load, controlnet=controlnet_model)

        try:
            if controlnet_model:
                logger.info("initializing_controlnet_pipeline")
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_model, torch_dtype=self.dtype, use_safetensors=True
                )
                # Note: This assumes SD 1.5 based models.
                # For SDXL, we would need StableDiffusionXLControlNetPipeline
                pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    model_to_load,
                    controlnet=controlnet,
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                    use_safetensors=True,
                    safety_checker=None,
                )
            else:
                pipeline = DiffusionPipeline.from_pretrained(
                    model_to_load,
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                    use_safetensors=True,
                )

            # Move pipeline to device
            pipeline = pipeline.to(self.device)

            # Use better scheduler (if compatible)
            try:
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config
                )
            except ValueError as e:
                # Some models have incompatible scheduler configs - use their default
                logger.warning("scheduler_override_skipped", reason=str(e))

            # Device-specific optimizations
            if self.device == "mps":
                logger.info(
                    "applying_mps_optimizations",
                    action="configuring_for_apple_silicon",
                    dtype=str(self.dtype),
                )
                # Don't use attention/VAE slicing on MPS - it's slower than direct computation

                # Log actual device placement
                logger.info(
                    "verifying_device_placement",
                    unet_device=(
                        str(pipeline.unet.device)
                        if hasattr(pipeline.unet, "device")
                        else "unknown"
                    ),
                    vae_device=(
                        str(pipeline.vae.device)
                        if hasattr(pipeline.vae, "device")
                        else "unknown"
                    ),
                    unet_dtype=(
                        str(pipeline.unet.dtype)
                        if hasattr(pipeline.unet, "dtype")
                        else "unknown"
                    ),
                    vae_dtype=(
                        str(pipeline.vae.dtype)
                        if hasattr(pipeline.vae, "dtype")
                        else "unknown"
                    ),
                )
            else:
                # Enable memory-efficient optimizations for CUDA/CPU
                logger.info("applying_memory_optimizations", device=self.device)

                # Enable attention slicing for memory efficiency
                try:
                    pipeline.enable_attention_slicing(1)
                    logger.debug("attention_slicing_enabled")
                except Exception as e:
                    logger.warning("attention_slicing_failed", error=str(e))

                # Enable VAE slicing for large images
                try:
                    pipeline.enable_vae_slicing()
                    logger.debug("vae_slicing_enabled")
                except Exception as e:
                    logger.warning("vae_slicing_failed", error=str(e))

                # Try to enable xformers memory efficient attention
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers_enabled", benefit="reduced_memory_usage")
                except Exception as e:
                    logger.debug(
                        "xformers_not_available",
                        hint="pip install xformers",
                        error=str(e),
                    )
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_slicing()

            # Assign to instance attribute after all setup is complete
            self.pipeline = pipeline

            # Cache the loaded model for future use
            self._model_cache[model_to_load] = pipeline
            self._current_model_id = model_to_load

            logger.info("model_loaded", model=model_to_load, device=self.device)
        except Exception as e:
            logger.error("model_load_failed", model=model_to_load, error=str(e))
            raise

    def load_refiner(
        self, refiner_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    ):
        """Load the refiner pipeline."""
        logger.info("loading_refiner", model=refiner_id)

        try:
            # Get text_encoder_2 and vae from pipeline if available
            text_encoder_2 = None
            vae = None
            if self.pipeline is not None:
                if hasattr(self.pipeline, "text_encoder_2"):
                    text_encoder_2 = self.pipeline.text_encoder_2
                if hasattr(self.pipeline, "vae"):
                    vae = self.pipeline.vae

            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_id,
                text_encoder_2=text_encoder_2,
                vae=vae,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None,
            )

            # Move refiner to device and assign to instance
            self.refiner = refiner.to(self.device)

            logger.info("refiner_loaded", model=refiner_id)
        except Exception as e:
            logger.error("refiner_load_failed", model=refiner_id, error=str(e))
            raise

    def load_lora(self, lora_path: Path, lora_scale: float = 0.8):
        """Load LoRA weights from trained model."""
        if not self.pipeline:
            raise RuntimeError("Load model first using load_model()")

        logger.info("loading_lora", path=str(lora_path), scale=lora_scale)

        try:
            # Load LoRA weights using diffusers method
            self.pipeline.load_lora_weights(str(lora_path))

            # Apply LoRA scale
            self.pipeline.fuse_lora(lora_scale=lora_scale)

            logger.info("lora_loaded", path=str(lora_path), scale=lora_scale)
        except Exception as e:
            logger.error("lora_load_failed", path=str(lora_path), error=str(e))
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: int | None = None,
        use_refiner: bool = False,
        control_image: Image.Image | None = None,
        controlnet_conditioning_scale: float = 1.0,
        on_progress: "Callable[[int, int, str], None] | None" = None,
        avoid_people: bool = True,
    ) -> list[Image.Image]:
        """Generate images from prompt.

        Args:
            prompt: The text prompt for generation
            negative_prompt: What to avoid in the generation
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt (7-9 recommended)
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            use_refiner: Whether to use the refiner model (if loaded)
            control_image: Optional PIL Image for ControlNet guidance (e.g. Canny edge map)
            controlnet_conditioning_scale: Strength of ControlNet guidance (0.0-1.0)
            on_progress: Optional callback(step, total_steps, message) for progress updates            avoid_people: If True, adds negative prompts to avoid generating people when not explicitly requested
        Returns:
            List of generated PIL images
        """
        if not self.pipeline:
            raise RuntimeError("Load model first using load_model()")

        # Add people avoidance to negative prompt if not explicitly wanted
        person_keywords = [
            "person",
            "people",
            "man",
            "woman",
            "men",
            "women",
            "human",
            "portrait",
            "face",
            "selfie",
        ]
        prompt_lower = prompt.lower()
        has_people_in_prompt = any(kw in prompt_lower for kw in person_keywords)

        if avoid_people and not has_people_in_prompt:
            # Add to negative prompt to avoid unwanted human generation
            people_negative = "person, people, human, portrait, face, man, woman"
            if negative_prompt:
                negative_prompt = f"{negative_prompt}, {people_negative}"
            else:
                negative_prompt = people_negative

        logger.info(
            "generating_images",
            prompt=prompt[:100],
            num_images=num_images,
            steps=num_inference_steps,
            use_refiner=use_refiner,
            has_control_image=bool(control_image),
            avoid_people=avoid_people,
        )

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Progress callback
        def progress_callback(step: int, timestep: int, latents):
            progress_pct = int((step / num_inference_steps) * 100)
            bar_length = 30
            filled = int(bar_length * step // num_inference_steps)
            bar = "\u2588" * filled + "\u2591" * (bar_length - filled)
            print(
                f"\r   [{bar}] {progress_pct}% ({step}/{num_inference_steps} steps)",
                end="",
                flush=True,
            )
            # Call external progress callback if provided (for WebSocket updates)
            if on_progress is not None:
                message = f"Generating: step {step}/{num_inference_steps}"
                on_progress(step, num_inference_steps, message)

        # If using refiner, we need to output latents from base
        output_type = "pil"
        if use_refiner and self.refiner:
            output_type = "latent"

        # Prepare kwargs
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "generator": generator,
            "callback": progress_callback,
            "callback_steps": 1,
            "output_type": output_type,
        }

        # Add ControlNet arguments if applicable
        if control_image is not None:
            # Check if pipeline supports controlnet
            if (
                hasattr(self.pipeline, "controlnet")
                or "ControlNet" in self.pipeline.__class__.__name__
            ):
                call_kwargs["image"] = control_image
                call_kwargs["controlnet_conditioning_scale"] = (
                    controlnet_conditioning_scale
                )
            else:
                logger.warning("control_image_ignored_pipeline_no_support")

        # Refiner denoising end
        if use_refiner and self.refiner:
            call_kwargs["denoising_end"] = 0.8

        result = self.pipeline(**call_kwargs)
        print()  # New line after progress bar

        images = result.images

        # Run Refiner if requested
        if use_refiner and self.refiner:
            logger.info("running_refiner")
            images = self.refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=max(10, int(num_inference_steps * 0.2)),
                denoising_start=0.8,
                image=images,
                generator=generator,
            ).images

        # Filter out black/blank/NaN images (common MPS issue)
        valid_images = []
        for i, img in enumerate(images):
            is_invalid, reason = is_black_or_blank(img)
            if is_invalid:
                logger.warning(
                    "rejecting_invalid_image",
                    index=i,
                    reason=reason,
                    device=self.device,
                )
            else:
                valid_images.append(img)

        # If all images were invalid, log error but return empty list
        if not valid_images and images:
            logger.error(
                "all_generated_images_invalid",
                original_count=len(images),
                device=self.device,
                hint="Try using float32 dtype on MPS or check model configuration",
            )

        logger.info(
            "images_generated",
            total=len(images),
            valid=len(valid_images),
            rejected=len(images) - len(valid_images),
        )

        # Return only valid images
        images = valid_images

        # Clear GPU cache after generation to prevent memory buildup
        self.clear_vram()

        return images

    def clear_vram(self):
        """Clear GPU memory cache to prevent memory leaks in long-running sessions.

        Should be called after each generation cycle, especially in 24/7 operation.
        This helps prevent VRAM fragmentation and out-of-memory errors.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("cuda_vram_cleared")
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.debug("mps_vram_cleared")

    def unload(self):
        """Unload model from memory and cleanup resources."""
        if self.refiner:
            logger.info("unloading_refiner")
            del self.refiner
            self.refiner = None

        # Clear model cache
        if self._model_cache:
            logger.info("clearing_model_cache", num_models=len(self._model_cache))
            for model_id in list(self._model_cache.keys()):
                del self._model_cache[model_id]
            self._model_cache.clear()
            self._current_model_id = None

        if self.pipeline:
            logger.info("unloading_model")
            del self.pipeline
            self.pipeline = None

            # Clear GPU cache
            self.clear_vram()

            logger.info("model_unloaded")
