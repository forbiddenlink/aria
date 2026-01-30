"""Automated image curation using CLIP."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..utils.logging import get_logger

logger = get_logger(__name__)


def is_black_or_blank(image: Image.Image, threshold: float = 0.02) -> tuple[bool, str]:
    """Detect if an image is black, blank, or contains NaN values.

    Args:
        image: PIL Image to check
        threshold: Maximum allowed mean pixel value ratio for "black" detection (0-1)

    Returns:
        Tuple of (is_invalid, reason) where is_invalid is True if image should be rejected
    """
    try:
        arr = np.array(image)

        # Check for NaN values (common MPS issue)
        if np.isnan(arr).any():
            logger.warning("detected_nan_image", has_nan=True)
            return True, "contains_nan_values"

        # Check for completely black image (max value is 0 or very low)
        if arr.max() == 0:
            logger.warning("detected_black_image", max_value=0)
            return True, "completely_black"

        # Check for nearly black image (mean is very low)
        mean_value = arr.mean() / 255.0  # Normalize to 0-1
        if mean_value < threshold:
            logger.warning("detected_nearly_black_image", mean_normalized=mean_value)
            return True, "nearly_black"

        # Check for uniform color (no variation - likely failed generation)
        std_value = arr.std()
        if std_value < 1.0:  # Very low standard deviation
            logger.warning("detected_uniform_image", std=std_value)
            return True, "uniform_color"

        # Check for mostly white/blown out
        white_ratio = np.sum(arr > 250) / arr.size
        if white_ratio > 0.95:
            logger.warning("detected_blown_out_image", white_ratio=white_ratio)
            return True, "blown_out"

        return False, "valid"

    except Exception as e:
        logger.error("image_validation_failed", error=str(e))
        return True, f"validation_error: {e}"


@dataclass
class QualityMetrics:
    """Image quality metrics."""

    aesthetic_score: float
    clip_score: float
    technical_score: float

    @property
    def overall_score(self) -> float:
        """Weighted average."""
        return (
            self.aesthetic_score * 0.5
            + self.clip_score * 0.3
            + self.technical_score * 0.2
        )


class ImageCurator:
    """CLIP-based image curation."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        # Note: CLIP loading is deferred to avoid import errors if not installed yet
        self.model = None
        self.preprocess = None
        logger.info("curator_initialized", device=device)

    def _load_clip(self):
        """Lazy load CLIP model."""
        if self.model is None:
            try:
                import clip

                self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
                logger.info("clip_model_loaded")
            except ImportError:
                logger.warning(
                    "clip_not_installed",
                    message="Install with: pip install git+https://github.com/openai/CLIP.git",
                )
                # Don't raise - allow graceful degradation
                return False
        return True

    def evaluate(self, image: Image.Image, prompt: str) -> QualityMetrics:
        """Evaluate image quality."""
        # Load CLIP if not already loaded - if it fails, return default scores
        if self.model is None and not self._load_clip():
            return QualityMetrics(
                aesthetic_score=0.5,
                clip_score=0.5,
                technical_score=0.5,
            )

        # CLIP score (text-image alignment)
        clip_score = self._compute_clip_score(image, prompt)

        # Aesthetic score (placeholder - implement with aesthetic predictor)
        aesthetic_score = self._estimate_aesthetic(image)

        # Technical score (resolution, sharpness)
        technical_score = self._compute_technical_score(image)

        metrics = QualityMetrics(
            aesthetic_score=aesthetic_score,
            clip_score=clip_score,
            technical_score=technical_score,
        )

        logger.info(
            "image_evaluated",
            overall=round(metrics.overall_score, 2),
            aesthetic=round(aesthetic_score, 2),
            clip=round(clip_score, 2),
        )

        return metrics

    def _compute_clip_score(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP similarity score."""
        import clip
        import torch

        assert self.model is not None and self.preprocess is not None
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            text_token = clip.tokenize([prompt]).to(self.device)

            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_token)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (image_features @ text_features.T).item()

        return float(max(0.0, similarity))  # Clip to [0, 1]

    def _estimate_aesthetic(self, image: Image.Image) -> float:
        """Estimate aesthetic score (placeholder)."""
        # TODO: Implement with LAION aesthetic predictor
        # For now, return a dummy score based on image properties
        width, height = image.size
        # Prefer images close to square aspect ratio
        aspect_ratio = min(width, height) / max(width, height)
        return float(0.5 + (aspect_ratio * 0.3))  # Range: 0.5-0.8

    def _detect_blur(self, image: Image.Image) -> float:
        """Detect image blur using Laplacian variance.
        
        Returns:
            float: Blur score (0-1, higher is sharper)
        """
        try:
            import cv2

            # Convert PIL to numpy array (grayscale for blur detection)
            img_array = np.array(image.convert('L'))
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            variance = laplacian.var()
            
            # Normalize: >100 is sharp, <50 is blurry
            # Scale to 0-1 range
            blur_score = min(variance / 100.0, 1.0)
            
            logger.debug(
                "blur_detection",
                variance=variance,
                score=blur_score,
                assessment="sharp" if blur_score > 0.7 else "moderate" if blur_score > 0.5 else "blurry"
            )
            
            return float(blur_score)
        except ImportError:
            logger.warning("opencv_not_installed", message="Install opencv-python for blur detection")
            return 0.8  # Default to moderate score if opencv not available
        except Exception as e:
            logger.error("blur_detection_failed", error=str(e))
            return 0.8
    
    def _detect_artifacts(self, image: Image.Image) -> float:
        """Detect compression artifacts and anomalies.
        
        Returns:
            float: Artifact score (0-1, higher is better/fewer artifacts)
        """
        try:
            img_array = np.array(image)
            
            # Check for extreme values (blown highlights/crushed shadows)
            # Images with >30% extreme pixels likely have issues
            extremes = np.sum((img_array < 10) | (img_array > 245))
            extreme_ratio = extremes / img_array.size
            
            # Check for color diversity (banding/posterization detection)
            # More unique colors = better (less banding)
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            color_diversity = min(unique_colors / (total_pixels * 0.1), 1.0)  # Expect 10% unique
            
            # Combine metrics (penalize extremes and low diversity)
            artifact_score = (1.0 - min(extreme_ratio * 3, 1.0)) * 0.5 + color_diversity * 0.5
            
            logger.debug(
                "artifact_detection",
                extreme_ratio=extreme_ratio,
                color_diversity=color_diversity,
                unique_colors=unique_colors,
                score=artifact_score,
                assessment="clean" if artifact_score > 0.7 else "moderate" if artifact_score > 0.5 else "artifacts"
            )
            
            return float(artifact_score)
        except Exception as e:
            logger.error("artifact_detection_failed", error=str(e))
            return 0.8  # Default to moderate score on error
    
    def _compute_technical_score(self, image: Image.Image) -> float:
        """Compute technical quality score with blur and artifact detection.
        
        Combines multiple technical metrics:
        - Resolution quality (40%)
        - Sharpness/blur (40%)
        - Artifact detection (20%)
        """
        # Check resolution
        width, height = image.size
        resolution = width * height
        resolution_score = min(1.0, resolution / (1024 * 1024))
        
        # Detect blur
        blur_score = self._detect_blur(image)
        
        # Detect artifacts
        artifact_score = self._detect_artifacts(image)
        
        # Weighted combination
        technical_score = (
            resolution_score * 0.4 +
            blur_score * 0.4 +
            artifact_score * 0.2
        )
        
        logger.debug(
            "technical_score_calculated",
            resolution=resolution_score,
            blur=blur_score,
            artifacts=artifact_score,
            total=technical_score
        )
        
        return float(technical_score)

    def should_keep(self, metrics: QualityMetrics, threshold: float = 0.6) -> bool:
        """Determine if image should be kept."""
        return metrics.overall_score >= threshold
