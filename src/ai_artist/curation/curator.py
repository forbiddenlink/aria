"""Automated image curation using CLIP."""

from dataclasses import dataclass

from PIL import Image

from ..utils.logging import get_logger

logger = get_logger(__name__)


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

    def _compute_technical_score(self, image: Image.Image) -> float:
        """Compute technical quality score."""
        # Check resolution
        width, height = image.size
        resolution_score = min(1.0, (width * height) / (1024 * 1024))

        # TODO: Add blur detection, artifact detection

        return float(resolution_score)

    def should_keep(self, metrics: QualityMetrics, threshold: float = 0.6) -> bool:
        """Determine if image should be kept."""
        return metrics.overall_score >= threshold
