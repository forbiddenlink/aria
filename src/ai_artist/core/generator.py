"""Image generation using Stable Diffusion + LoRA."""

from pathlib import Path
from typing import Literal

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

    def load_model(self, controlnet_model: str | None = None):
        """Load the diffusion pipeline."""
        logger.info("loading_model", model=self.model_id, controlnet=controlnet_model)

        try:
            if controlnet_model:
                logger.info("initializing_controlnet_pipeline")
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_model, torch_dtype=self.dtype, use_safetensors=True
                )
                # Note: This assumes SD 1.5 based models.
                # For SDXL, we would need StableDiffusionXLControlNetPipeline
                pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    self.model_id,
                    controlnet=controlnet,
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                    use_safetensors=True,
                    safety_checker=None,
                )
            else:
                pipeline = DiffusionPipeline.from_pretrained(
                    self.model_id,
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
                # Enable memory-efficient slicing for CUDA/CPU
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_slicing()

            # Assign to instance attribute after all setup is complete
            self.pipeline = pipeline

            logger.info("model_loaded", model=self.model_id, device=self.device)
        except Exception as e:
            logger.error("model_load_failed", model=self.model_id, error=str(e))
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

        Returns:
            List of generated PIL images
        """
        if not self.pipeline:
            raise RuntimeError("Load model first using load_model()")

        logger.info(
            "generating_images",
            prompt=prompt[:100],
            num_images=num_images,
            steps=num_inference_steps,
            use_refiner=use_refiner,
            has_control_image=bool(control_image),
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

        # Clear MPS cache after generation to prevent memory buildup
        if self.device == "mps":
            torch.mps.empty_cache()

        return images

    def unload(self):
        """Unload model from memory and cleanup resources."""
        if self.refiner:
            logger.info("unloading_refiner")
            del self.refiner
            self.refiner = None

        if self.pipeline:
            logger.info("unloading_model")
            del self.pipeline
            self.pipeline = None

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                # MPS cleanup
                torch.mps.empty_cache()

            logger.info("model_unloaded")
