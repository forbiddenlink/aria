"""Tests for gallery manager."""

import json

import pytest
from PIL import Image

from src.ai_artist.gallery.manager import GalleryManager


@pytest.fixture
def test_gallery(tmp_path):
    """Create test gallery."""
    return GalleryManager(tmp_path / "test_gallery")


def test_save_image(test_gallery):
    """Test saving image with metadata."""
    # Create a test image
    image = Image.new("RGB", (512, 512), color="red")
    prompt = "a test image"
    metadata = {"seed": 42, "model": "test-model"}

    # Save image
    saved_path = test_gallery.save_image(
        image=image, prompt=prompt, metadata=metadata, featured=False
    )

    # Verify file exists
    assert saved_path.exists()
    assert saved_path.suffix == ".png"

    # Verify metadata file exists
    metadata_path = saved_path.with_suffix(".json")
    assert metadata_path.exists()

    # Verify metadata content
    saved_metadata = json.loads(metadata_path.read_text())
    assert saved_metadata["prompt"] == prompt
    assert saved_metadata["metadata"]["seed"] == 42
    assert saved_metadata["featured"] is False


def test_save_featured_image(test_gallery):
    """Test saving featured image to correct directory."""
    image = Image.new("RGB", (512, 512), color="blue")
    prompt = "featured test"
    metadata = {"seed": 123}

    saved_path = test_gallery.save_image(
        image=image, prompt=prompt, metadata=metadata, featured=True
    )

    # Verify it's in the featured directory
    assert "featured" in str(saved_path)


def test_list_images(test_gallery):
    """Test listing images in gallery."""
    # Create some test images
    for i in range(3):
        image = Image.new("RGB", (512, 512), color="red")
        test_gallery.save_image(
            image=image,
            prompt=f"test {i}",
            metadata={"seed": i},
            featured=(i == 0),
        )

    # List all images
    all_images = test_gallery.list_images(featured_only=False)
    assert len(all_images) >= 3

    # List only featured
    featured_images = test_gallery.list_images(featured_only=True)
    assert len(featured_images) >= 1
