"""Unit tests for the curation system."""

from unittest.mock import patch

import pytest
from ai_artist.curation.curator import ImageCurator, QualityMetrics
from PIL import Image


def test_quality_metrics_overall_score():
    """Test that overall score is calculated correctly."""
    metrics = QualityMetrics(
        aesthetic_score=0.8,
        clip_score=0.7,
        technical_score=0.9,
    )

    # 0.8*0.5 + 0.7*0.3 + 0.9*0.2 = 0.4 + 0.21 + 0.18 = 0.79
    expected = 0.79
    assert abs(metrics.overall_score - expected) < 0.01


def test_curator_initialization():
    """Test curator initializes correctly."""
    curator = ImageCurator(device="cpu")
    assert curator.device == "cpu"
    assert curator.model is None  # Lazy loaded


@pytest.mark.skipif(True, reason="Requires CLIP model download")
def test_curator_loads_clip():
    """Test that CLIP model loads when needed."""
    curator = ImageCurator(device="cpu")
    curator._load_clip()

    assert curator.model is not None
    assert curator.preprocess is not None


def test_curator_evaluate_with_mock():
    """Test image evaluation with mocked CLIP."""
    curator = ImageCurator(device="cpu")

    # Create a test image
    img = Image.new("RGB", (512, 512), color="blue")
    prompt = "blue sky"

    with (
        patch.object(curator, "_load_clip"),
        patch.object(curator, "_compute_clip_score", return_value=0.8),
        patch.object(curator, "_estimate_aesthetic", return_value=0.7),
        patch.object(curator, "_compute_technical_score", return_value=0.9),
    ):

        metrics = curator.evaluate(img, prompt)

        assert metrics.clip_score == 0.8
        assert metrics.aesthetic_score == 0.7
        assert metrics.technical_score == 0.9
        assert 0 <= metrics.overall_score <= 1


def test_curator_should_keep():
    """Test quality threshold filtering."""
    curator = ImageCurator(device="cpu")

    # High quality image
    good_metrics = QualityMetrics(
        aesthetic_score=0.8,
        clip_score=0.75,
        technical_score=0.85,
    )
    assert curator.should_keep(good_metrics, threshold=0.6)

    # Low quality image
    bad_metrics = QualityMetrics(
        aesthetic_score=0.3,
        clip_score=0.2,
        technical_score=0.4,
    )
    assert not curator.should_keep(bad_metrics, threshold=0.6)


def test_aesthetic_score_aspect_ratio():
    """Test that aesthetic score considers aspect ratio."""
    curator = ImageCurator(device="cpu")

    # Square image (perfect aspect ratio)
    square_img = Image.new("RGB", (512, 512))
    score_square = curator._estimate_aesthetic(square_img)

    # Wide image (worse aspect ratio)
    wide_img = Image.new("RGB", (1024, 256))
    score_wide = curator._estimate_aesthetic(wide_img)

    # Square should score higher
    assert score_square > score_wide


def test_technical_score_resolution():
    """Test that technical score considers resolution."""
    curator = ImageCurator(device="cpu")

    # High resolution
    large_img = Image.new("RGB", (2048, 2048))
    score_large = curator._compute_technical_score(large_img)

    # Low resolution
    small_img = Image.new("RGB", (256, 256))
    score_small = curator._compute_technical_score(small_img)

    # Higher resolution should score higher
    assert score_large > score_small
