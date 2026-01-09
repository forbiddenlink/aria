"""Image generation using Stable Diffusion + LoRA."""

from pathlib import Path
from typing import Literal

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ImageGenerator:
    """Stable Diffusion image generator with LoRA support."""

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

        logger.info("initializing_generator", model=model_id, device=device)

    def load_model(self):
        """Load the diffusion pipeline."""
        logger.info("loading_model", model=self.model_id)

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
            use_safetensors=True,
        )

        # Optimizations
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()

        # Use better scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Fix for MPS dtype issues: ensure VAE uses float32 entirely
        if self.device == "mps":
            logger.info("applying_mps_fixes", action="converting_vae_to_float32")
            # Convert entire VAE to float32 to avoid dtype mismatches on MPS
            self.pipeline.vae = self.pipeline.vae.to(dtype=torch.float32)
            # Disable attention slicing on MPS as it can cause issues  
            self.pipeline.disable_attention_slicing()

        logger.info("model_loaded", model=self.model_id)

    def load_lora(self, lora_path: Path, lora_scale: float = 0.8):
        """Load LoRA weights."""
        if not self.pipeline:
            raise RuntimeError("Load model first using load_model()")

        logger.info("loading_lora", path=str(lora_path), scale=lora_scale)

        self.pipeline.load_lora_weights(str(lora_path))
        self.pipeline.fuse_lora(lora_scale=lora_scale)

        logger.info("lora_loaded", path=str(lora_path))

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
    ) -> list[Image.Image]:
        """Generate images from prompt."""
        if not self.pipeline:
            raise RuntimeError("Load model first using load_model()")

        logger.info(
            "generating_images",
            prompt=prompt[:100],
            num_images=num_images,
            steps=num_inference_steps,
        )

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )
        
        # Check for NaN values in generated images (common MPS issue)
        images = result.images
        for i, img in enumerate(images):
            import numpy as np
            arr = np.array(img)
            if np.isnan(arr).any() or arr.max() == 0:
                logger.warning(
                    "detected_nan_or_black_image",
                    index=i,
                    has_nan=np.isnan(arr).any(),
                    is_black=arr.max() == 0,
                    device=self.device
                )

        logger.info("images_generated", count=len(images))

        return images

    def unload(self):
        """Unload model from memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("model_unloaded")

