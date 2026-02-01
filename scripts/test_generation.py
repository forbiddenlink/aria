#!/usr/bin/env python3
"""Quick test script to verify image generation works."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.ai_artist.core.generator import ImageGenerator
from src.ai_artist.gallery.manager import GalleryManager
from src.ai_artist.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def main():
    """Run a quick generation test."""
    logger.info("starting_generation_test")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("using_cuda", gpu=torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("using_mps")
    else:
        device = "cpu"
        logger.info("using_cpu")

    # Use a smaller, faster model for testing
    # Note: SDXL Turbo can have issues on MPS, using regular SD 1.5 for better compatibility
    if device == "mps":
        model_id = "runwayml/stable-diffusion-v1-5"
        num_steps = 20
        logger.info("using_sd15_for_mps_compatibility", model=model_id)
    else:
        model_id = "stabilityai/sdxl-turbo"
        num_steps = 4
        logger.info("loading_model", model=model_id)

    # Initialize generator
    generator = ImageGenerator(
        model_id=model_id,
        device=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )

    try:
        generator.load_model()

        # Generate test image
        logger.info("generating_test_image")
        images = generator.generate(
            prompt="a beautiful sunset over mountains, highly detailed, artistic",
            num_images=1,
            width=512,
            height=512,
            num_inference_steps=num_steps,
            seed=42,
        )

        # Save to gallery
        gallery = GalleryManager(Path("gallery/test"))
        saved_path = gallery.save_image(
            image=images[0],
            prompt="test generation",
            metadata={"model": model_id, "seed": 42},
        )

        logger.info("test_complete", path=str(saved_path))
        print(f"\n✅ Test successful! Image saved to: {saved_path}")
        print("You can view the image in the gallery/test directory")

    except Exception as e:
        logger.error("test_failed", error=str(e), exc_info=True)
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    finally:
        generator.unload()


if __name__ == "__main__":
    main()
