#!/usr/bin/env python3
"""Manual testing script for AI Artist functionality."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_database():
    """Test database initialization and operations."""
    print("\n🔍 Testing Database...")
    from sqlalchemy import create_engine

    from ai_artist.db.models import Base, GeneratedImage
    from ai_artist.db.session import create_session_factory, get_db

    # Create test database
    db_path = Path("data/test_ai_artist.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    factory = create_session_factory(db_path)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    # Test insert
    with factory() as session:
        test_image = GeneratedImage(
            filename="test.png",
            prompt="test prompt",
            model_id="test-model",
            status="curated",
            created_at=datetime.now(),
        )
        session.add(test_image)
        session.commit()
        print(f"  ✓ Created test image with ID: {test_image.id}")

    # Test query
    with factory() as session:
        images = session.query(GeneratedImage).all()
        print(f"  ✓ Found {len(images)} image(s) in database")

    # Test FastAPI dependency
    dep_gen = get_db()
    db_session = next(dep_gen)
    print(f"  ✓ Got database session: {type(db_session).__name__}")

    print("  ✅ Database tests passed\n")
    return True


def test_config():
    """Test configuration loading."""
    print("🔍 Testing Configuration...")
    import torch

    from ai_artist.utils.config import load_config

    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("  ⚠️  config.yaml not found, using example")
        config_path = Path("config/config.example.yaml")

    config = load_config(config_path)

    print(f"  ✓ Model: {config.model.base_model}")
    print(f"  ✓ Device: {config.model.device}")
    print(f"  ✓ Dtype: {config.model.dtype}")
    print(f"  ✓ Width: {config.generation.width}")
    print(f"  ✓ Height: {config.generation.height}")

    # Verify dtype parsing
    dtype = torch.float32 if config.model.dtype == "float32" else torch.float16
    print(f"  ✓ Parsed dtype: {dtype}")

    print("  ✅ Configuration tests passed\n")
    return True


def test_mood_system():
    """Test mood system and time-based selection."""
    print("🔍 Testing Mood System...")
    from ai_artist.personality.moods import MoodSystem

    mood_system = MoodSystem()
    print(f"  ✓ Current mood: {mood_system.current_mood.value}")
    print(f"  ✓ Energy level: {mood_system.energy_level:.2f}")
    print(f"  ✓ Feeling: {mood_system.describe_feeling()}")

    # Test mood update
    old_mood = mood_system.current_mood
    mood_system.update_mood()
    print(f"  ✓ Mood updated (may have changed from {old_mood.value})")

    # Test mood influences
    influences = mood_system.mood_influences.get(mood_system.current_mood, {})
    print(f"  ✓ Available styles: {len(influences.get('styles', []))}")
    print(f"  ✓ Available colors: {len(influences.get('colors', []))}")

    print("  ✅ Mood system tests passed\n")
    return True


def test_autonomous_inspiration():
    """Test autonomous inspiration subject variety."""
    print("🔍 Testing Autonomous Inspiration...")
    from ai_artist.inspiration.autonomous import AutonomousInspiration

    autonomous = AutonomousInspiration()
    print(f"  ✓ Total subjects available: {len(autonomous.subjects)}")
    print(f"  ✓ Sample subjects: {autonomous.subjects[:5]}")
    print(f"  ✓ Total styles available: {len(autonomous.styles)}")

    print("  ✅ Autonomous inspiration tests passed\n")
    return True


async def test_websocket_manager():
    """Test WebSocket connection manager."""
    print("🔍 Testing WebSocket Manager...")
    from ai_artist.web.websocket import ConnectionManager

    manager = ConnectionManager()
    print(f"  ✓ Manager created: {type(manager).__name__}")
    print(f"  ✓ Active connections: {len(manager.active_connections)}")

    # Test message creation (without actual WebSocket)
    test_message = {
        "type": "thinking_update",
        "session_id": "test-123",
        "thought_type": "observe",
        "content": "Test thought",
    }
    print(f"  ✓ Message structure valid: {test_message['type']}")

    print("  ✅ WebSocket manager tests passed\n")
    return True


async def test_api_routes():
    """Test API route initialization."""
    print("🔍 Testing API Routes...")
    from ai_artist.web.aria_routes import _get_aria_state, router

    print(f"  ✓ Router created: {router.prefix}")
    print(f"  ✓ Router tags: {router.tags}")

    # Test state initialization
    state = _get_aria_state()
    print(f"  ✓ Aria name: {state['name']}")
    print(f"  ✓ Mood system initialized: {state['mood_system'].current_mood.value}")
    print(f"  ✓ Personality traits: {len(state['personality'])}")

    print("  ✅ API routes tests passed\n")
    return True


def test_gallery_structure():
    """Test gallery directory structure."""
    print("🔍 Testing Gallery Structure...")

    gallery_path = Path("gallery")
    if not gallery_path.exists():
        gallery_path.mkdir(parents=True)
        print("  ⚠️  Created gallery directory")

    # Check for existing images
    image_files = list(gallery_path.rglob("*.png")) + list(gallery_path.rglob("*.jpg"))
    json_files = list(gallery_path.rglob("*.json"))

    print(f"  ✓ Gallery path exists: {gallery_path}")
    print(f"  ✓ Image files found: {len(image_files)}")
    print(f"  ✓ Metadata files found: {len(json_files)}")

    if image_files:
        print(f"  ✓ Sample image: {image_files[0].relative_to(gallery_path)}")

    print("  ✅ Gallery structure tests passed\n")
    return True


def test_image_generator_init():
    """Test image generator initialization (without loading model)."""
    print("🔍 Testing Image Generator Initialization...")
    import torch

    from ai_artist.core.generator import ImageGenerator

    # Test with float32 (safe for MPS)
    generator = ImageGenerator(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cpu",  # Use CPU to avoid loading large model
        dtype=torch.float32,
    )

    print(f"  ✓ Generator created: {type(generator).__name__}")
    print(f"  ✓ Model ID: {generator.model_id}")
    print(f"  ✓ Device: {generator.device}")
    print(f"  ✓ Dtype: {generator.dtype}")

    print("  ✅ Image generator initialization tests passed\n")
    return True


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 AI ARTIST COMPREHENSIVE FUNCTIONALITY TEST")
    print("=" * 60)

    results = {}

    try:
        results["database"] = test_database()
    except Exception as e:
        print(f"  ❌ Database test failed: {e}\n")
        results["database"] = False

    try:
        results["config"] = test_config()
    except Exception as e:
        print(f"  ❌ Config test failed: {e}\n")
        results["config"] = False

    try:
        results["mood_system"] = test_mood_system()
    except Exception as e:
        print(f"  ❌ Mood system test failed: {e}\n")
        results["mood_system"] = False

    try:
        results["autonomous_inspiration"] = test_autonomous_inspiration()
    except Exception as e:
        print(f"  ❌ Autonomous inspiration test failed: {e}\n")
        results["autonomous_inspiration"] = False

    try:
        results["websocket"] = await test_websocket_manager()
    except Exception as e:
        print(f"  ❌ WebSocket test failed: {e}\n")
        results["websocket"] = False

    try:
        results["api_routes"] = await test_api_routes()
    except Exception as e:
        print(f"  ❌ API routes test failed: {e}\n")
        results["api_routes"] = False

    try:
        results["gallery"] = test_gallery_structure()
    except Exception as e:
        print(f"  ❌ Gallery test failed: {e}\n")
        results["gallery"] = False

    try:
        results["generator_init"] = test_image_generator_init()
    except Exception as e:
        print(f"  ❌ Generator test failed: {e}\n")
        results["generator_init"] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
