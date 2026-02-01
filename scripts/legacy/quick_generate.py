#!/usr/bin/env python3
"""Quick single image generation test."""

import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_artist.core.generator import ImageGenerator
from src.ai_artist.gallery.manager import GalleryManager
from src.ai_artist.utils.logging import configure_logging

configure_logging()

# Use CPU for stable generation (MPS is slow on M1)
device = "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
print("🖥️  Using CPU for stable generation (MPS too slow)")

generator = ImageGenerator(
    model_id=model_id,
    device=device,
    dtype=torch.float32,
)

generator.load_model()

print("\n🎨 Generating test image...")

prompt = (
    "a beautiful sunset over mountains, golden hour, dramatic clouds, photorealistic"
)
images = generator.generate(
    prompt=prompt,
    num_images=1,
    num_inference_steps=15 if device == "cpu" else (20 if device == "mps" else 4),
)

# Save to gallery
print("\n💾 Saving to gallery...")
gallery = GalleryManager(gallery_dir=Path("gallery"))
saved_paths = []

for img in images:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sunset_{timestamp}.png"
    filepath = gallery.gallery_dir / filename
    img.save(filepath)
    saved_paths.append(filepath)
    print(f"   💾 Saved: {filename}")

print(f"\n✅ Success! Generated and saved {len(images)} image(s)")
print(f"📁 Gallery directory: {gallery.gallery_dir}")
print("\n🌐 View in gallery: http://localhost:8000")
print("\n🌐 View in gallery: http://localhost:8000")
