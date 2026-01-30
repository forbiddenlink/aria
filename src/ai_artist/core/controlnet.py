"""ControlNet helper utilities."""

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel
from PIL import Image

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ControlNetPreprocessor:
    """Helper for ControlNet image preprocessing."""

    @staticmethod
    def get_canny_image(image: Image.Image, low_threshold=100, high_threshold=200) -> Image.Image:
        """Convert a PIL image to a Canny edge map."""
        try:
            image_np = np.array(image)
            
            # Ensure 3 channels
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            image_np = cv2.Canny(image_np, low_threshold, high_threshold)
            image_np = image_np[:, :, None]
            image_np = np.concatenate([image_np, image_np, image_np], axis=2)
            canny_image = Image.fromarray(image_np)
            return canny_image
        except ImportError:
            logger.error("opencv_missing", action="install_opencv_python")
            raise RuntimeError("opencv-python is required for ControlNet Canny preprocessing")
        except Exception as e:
            logger.error("canny_preprocessing_failed", error=str(e))
            raise


class ControlNetLoader:
    """Helper to load ControlNet models."""

    @staticmethod
    def load(model_id: str, dtype: torch.dtype = torch.float16) -> ControlNetModel:
        """Load a ControlNet model."""
        logger.info("loading_controlnet", model=model_id)
        return ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True
        )
