"""
Ensemble curation using multiple models for better quality assessment.

Uses voting from multiple evaluation models to get more robust
quality scores and reduce bias from any single model.
"""

from dataclasses import dataclass

import structlog
from PIL import Image

from ai_artist.utils.config import Config

logger = structlog.get_logger(__name__)


@dataclass
class EnsembleVote:
    """Votes from ensemble of evaluation models."""

    clip_scores: list[float]
    aesthetic_scores: list[float]
    avg_clip: float
    avg_aesthetic: float
    consensus_strength: float  # 0-1, how much models agree
    final_score: float


class EnsembleCurator:
    """
    Multi-model ensemble for robust quality assessment.

    Uses multiple evaluation approaches and voting to reduce
    bias and get more reliable quality scores.
    """

    def __init__(self, config: Config):
        """Initialize ensemble curator."""
        self.config = config
        logger.info("ensemble_curator_initialized")

    async def evaluate_with_ensemble(
        self, image: Image.Image, prompt: str
    ) -> EnsembleVote:
        """
        Evaluate image quality using ensemble of models.

        Args:
            image: Image to evaluate
            prompt: Generation prompt

        Returns:
            Ensemble voting results
        """
        # Import curator for individual evaluations
        from ai_artist.curation.curator import ImageCurator

        curator = ImageCurator(self.config.model.base_model)

        # Get evaluation from primary curator
        primary_metrics = curator.evaluate(image, prompt)

        # In future, add more evaluation models here:
        # - Different CLIP models (OpenCLIP variants)
        # - Different aesthetic predictors
        # - Custom trained quality models
        # - Perceptual quality metrics (LPIPS, FID)

        # For now, use single evaluation but structure for expansion
        clip_scores = [primary_metrics.clip_score]
        aesthetic_scores = [primary_metrics.aesthetic_score]

        # Calculate consensus (for single model, it's 1.0)
        consensus = 1.0

        # Average scores
        avg_clip = sum(clip_scores) / len(clip_scores)
        avg_aesthetic = sum(aesthetic_scores) / len(aesthetic_scores)

        # Weighted final score
        final_score = 0.6 * avg_clip + 0.4 * avg_aesthetic

        vote = EnsembleVote(
            clip_scores=clip_scores,
            aesthetic_scores=aesthetic_scores,
            avg_clip=avg_clip,
            avg_aesthetic=avg_aesthetic,
            consensus_strength=consensus,
            final_score=final_score,
        )

        logger.debug(
            "ensemble_evaluation_complete",
            final_score=final_score,
            consensus=consensus,
        )

        return vote

    async def evaluate_batch_with_ensemble(
        self, images: list[Image.Image], prompt: str
    ) -> list[EnsembleVote]:
        """
        Batch evaluate multiple images with ensemble.

        Args:
            images: Images to evaluate
            prompt: Generation prompt

        Returns:
            List of ensemble votes
        """
        from ai_artist.curation.curator import ImageCurator

        curator = ImageCurator(self.config.model.base_model)

        # Use batch evaluation for speed
        metrics_list = curator.evaluate_batch(images, prompt)

        # Convert to ensemble votes
        votes = []
        for metrics in metrics_list:
            vote = EnsembleVote(
                clip_scores=[metrics.clip_score],
                aesthetic_scores=[metrics.aesthetic_score],
                avg_clip=metrics.clip_score,
                avg_aesthetic=metrics.aesthetic_score,
                consensus_strength=1.0,  # Single model
                final_score=0.6 * metrics.clip_score + 0.4 * metrics.aesthetic_score,
            )
            votes.append(vote)

        logger.info("batch_ensemble_evaluation_complete", count=len(images))
        return votes

    def rank_by_ensemble(
        self, images: list[Image.Image], votes: list[EnsembleVote]
    ) -> list[tuple[Image.Image, EnsembleVote]]:
        """
        Rank images by ensemble voting results.

        Args:
            images: Images to rank
            votes: Corresponding ensemble votes

        Returns:
            List of (image, vote) tuples sorted by quality
        """
        paired = list(zip(images, votes, strict=False))
        # Sort by final score (highest first)
        ranked = sorted(paired, key=lambda x: x[1].final_score, reverse=True)

        logger.debug(
            "ensemble_ranking_complete",
            top_score=ranked[0][1].final_score if ranked else 0,
        )

        return ranked
