"""Tests for main AIArtist class."""

from unittest.mock import MagicMock, patch

import pytest
from ai_artist.main import AIArtist


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = MagicMock()
    config.model.base_model = "test/model"
    config.model.device = "cpu"
    config.model.dtype = "float32"
    config.model.lora_path = None
    config.model.lora_scale = 0.8
    config.api_keys.unsplash_access_key = "test_key"
    config.generation.width = 512
    config.generation.height = 512
    config.generation.num_inference_steps = 10
    config.generation.guidance_scale = 7.5
    config.generation.num_images = 1
    config.curation.enabled = True
    return config


class TestAIArtist:
    """Test AIArtist class."""

    @patch("ai_artist.main.CreationScheduler")
    @patch("ai_artist.main.ImageCurator")
    @patch("ai_artist.main.UnsplashClient")
    @patch("ai_artist.main.GalleryManager")
    @patch("ai_artist.main.ImageGenerator")
    @patch("ai_artist.main.configure_logging")
    def test_init(
        self,
        mock_logging,
        mock_generator_class,
        mock_gallery,
        mock_unsplash,
        mock_curator,
        mock_scheduler,
        mock_config,
    ):
        """Test AIArtist initialization."""
        # Mock generator instance
        mock_gen = MagicMock()
        mock_generator_class.return_value = mock_gen

        artist = AIArtist(config=mock_config)

        # Verify components were created
        assert artist.config == mock_config
        assert artist.generator is not None
        assert artist.gallery is not None
        assert artist.unsplash is not None
        assert artist.curator is not None
        assert artist.scheduler is not None

        # Verify load_model was called
        mock_gen.load_model.assert_called_once()

    @patch("ai_artist.main.CreationScheduler")
    @patch("ai_artist.main.ImageCurator")
    @patch("ai_artist.main.UnsplashClient")
    @patch("ai_artist.main.GalleryManager")
    @patch("ai_artist.main.ImageGenerator")
    @patch("ai_artist.main.configure_logging")
    def test_init_with_lora(
        self,
        mock_logging,
        mock_generator_class,
        mock_gallery,
        mock_unsplash,
        mock_curator,
        mock_scheduler,
        mock_config,
        tmp_path,
    ):
        """Test initialization with LoRA."""
        # Create mock LoRA path
        lora_path = tmp_path / "test_lora"
        lora_path.mkdir()
        mock_config.model.lora_path = str(lora_path)

        # Mock generator instance
        mock_gen = MagicMock()
        mock_generator_class.return_value = mock_gen

        _ = AIArtist(config=mock_config)

        # Verify load_lora was called
        mock_gen.load_lora.assert_called_once()
