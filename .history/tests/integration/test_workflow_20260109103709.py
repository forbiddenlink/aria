"""Integration tests for the full artwork creation workflow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ai_artist.main import AIArtist
from ai_artist.utils.config import Config


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=Config)
    config.model.base_model = "runwayml/stable-diffusion-v1-5"
    config.model.device = "cpu"  # Use CPU for tests
    config.model.dtype = "float32"
    config.generation.width = 256  # Smaller for faster tests
    config.generation.height = 256
    config.generation.num_inference_steps = 5  # Fewer steps
    config.generation.guidance_scale = 7.5
    config.generation.num_variations = 2
    config.generation.negative_prompt = "blurry"
    config.api_keys.unsplash_access_key = "test_key"
    return config


@pytest.fixture
def mock_unsplash_photo():
    """Mock Unsplash API response."""
    return {
        "id": "test_photo_123",
        "description": "A beautiful landscape",
        "alt_description": "mountain sunset",
        "urls": {"regular": "https://example.com/photo.jpg"},
        "links": {"download_location": "https://api.unsplash.com/download/123"},
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_artwork_creation_workflow(
    mock_config, mock_unsplash_photo, tmp_path
):
    """Test the complete artwork creation workflow."""
    # Mock the generator to avoid actual model loading
    with (
        patch("ai_artist.main.ImageGenerator") as mock_gen_class,
        patch("ai_artist.main.UnsplashClient") as mock_unsplash_class,
        patch("ai_artist.main.ImageCurator") as mock_curator_class,
        patch("ai_artist.main.GalleryManager") as mock_gallery_class,
    ):

        # Setup mocks
        mock_generator = MagicMock()
        mock_generator.generate.return_value = [MagicMock(), MagicMock()]  # 2 images
        mock_gen_class.return_value = mock_generator

        mock_unsplash = AsyncMock()
        mock_unsplash.get_random_photo.return_value = mock_unsplash_photo
        mock_unsplash.trigger_download.return_value = None
        mock_unsplash_class.return_value = mock_unsplash

        mock_curator = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.overall_score = 0.75
        mock_metrics.aesthetic_score = 0.8
        mock_metrics.clip_score = 0.7
        mock_curator.evaluate.return_value = mock_metrics
        mock_curator_class.return_value = mock_curator

        mock_gallery = MagicMock()
        mock_gallery.save_image.return_value = tmp_path / "test_image.png"
        mock_gallery_class.return_value = mock_gallery

        # Create app instance
        app = AIArtist(mock_config)
        app.initialize()

        # Test artwork creation
        result = await app.create_artwork(theme="test landscape")

        # Verify workflow
        assert mock_unsplash.get_random_photo.called
        assert mock_generator.generate.called
        assert mock_curator.evaluate.call_count == 2  # Evaluated both images
        assert mock_gallery.save_image.called
        assert result == tmp_path / "test_image.png"


@pytest.mark.integration
def test_configuration_loading(tmp_path):
    """Test that configuration loads correctly."""
    from ai_artist.utils.config import load_config

    # Create a test config file
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        """
model:
  base_model: "runwayml/stable-diffusion-v1-5"
  device: "cpu"
  dtype: "float32"

generation:
  width: 512
  height: 512
  num_inference_steps: 30
  guidance_scale: 7.5
  num_variations: 3
  negative_prompt: "blurry, low quality"

api_keys:
  unsplash_access_key: "test_access_key"
"""
    )

    config = load_config(config_path)

    assert config.model.base_model == "runwayml/stable-diffusion-v1-5"
    assert config.model.device == "cpu"
    assert config.generation.width == 512
    assert config.generation.num_variations == 3
    assert config.api_keys.unsplash_access_key == "test_access_key"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_handling_api_failure(mock_config):
    """Test that the app handles API failures gracefully."""
    with (
        patch("ai_artist.main.ImageGenerator") as mock_gen_class,
        patch("ai_artist.main.UnsplashClient") as mock_unsplash_class,
    ):

        # Setup mocks
        mock_generator = MagicMock()
        mock_gen_class.return_value = mock_generator

        mock_unsplash = AsyncMock()
        mock_unsplash.get_random_photo.side_effect = Exception("API Error")
        mock_unsplash_class.return_value = mock_unsplash

        # Create app instance
        app = AIArtist(mock_config)
        app.initialize()

        # Test that error is raised
        with pytest.raises(Exception, match="API Error"):
            await app.create_artwork(theme="test")
