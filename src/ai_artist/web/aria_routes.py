"""Aria API routes - personality, state, and creation endpoints."""

import asyncio
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..personality.enhanced_memory import EnhancedMemorySystem
from ..personality.moods import Mood, MoodSystem
from ..personality.profile import ArtisticProfile
from ..utils.config import load_config
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
router = APIRouter(prefix="/api/aria", tags=["aria"])

# Singleton state for Aria (initialized on first request)
_aria_state: dict[str, Any] | None = None


class AriaStateResponse(BaseModel):
    """Aria's current state."""

    name: str
    mood: str
    mood_intensity: float = 0.7
    energy: float
    feeling: str
    paintings_created: int
    personality: dict[str, float]
    portfolio: list[dict] | None = None
    experience: dict[str, Any] | None = None
    style_axes: dict[str, float] | None = None


class AriaCreateResponse(BaseModel):
    """Response from creation endpoint."""

    success: bool
    subject: str | None = None
    style: str | None = None
    prompt: str | None = None
    reflection: str | None = None
    thinking: str | None = None
    critique_history: list[dict] | None = None
    image_url: str | None = None
    session_id: str | None = None
    error: str | None = None


class AriaStatementResponse(BaseModel):
    """Artist statement response."""

    statement: str
    name: str


def _get_aria_state() -> dict[str, Any]:
    """Get or initialize Aria's state."""
    global _aria_state

    if _aria_state is None:
        # Initialize fresh state
        _aria_state = {
            "name": "Aria",
            "mood_system": MoodSystem(),
            "memory": EnhancedMemorySystem(),
            "profile": ArtisticProfile(name="Aria"),
            "paintings_created": 0,
            "portfolio": [],
            # OCEAN personality traits
            "personality": {
                "openness": random.uniform(0.6, 0.9),  # High - creative
                "conscientiousness": random.uniform(0.4, 0.7),
                "extraversion": random.uniform(0.3, 0.7),
                "agreeableness": random.uniform(0.3, 0.6),
                "neuroticism": random.uniform(0.4, 0.7),  # Artistic temperament
            },
        }
        logger.info(
            "aria_state_initialized",
            mood=_aria_state["mood_system"].current_mood.value,
        )

    return _aria_state


def _load_portfolio_from_gallery() -> list[dict]:
    """Load portfolio from gallery directory."""
    portfolio: list[dict] = []
    gallery_path = Path("gallery")

    if not gallery_path.exists():
        return portfolio

    # Find all JSON metadata files
    for json_file in gallery_path.rglob("*.json"):
        try:
            import json

            with open(json_file) as f:
                metadata = json.load(f)

            # Find corresponding image
            image_path = json_file.with_suffix(".png")
            if not image_path.exists():
                image_path = json_file.with_suffix(".jpg")

            if image_path.exists():
                portfolio.append(
                    {
                        "number": len(portfolio),
                        "subject": metadata.get("prompt", "").split(",")[0][:50],
                        "prompt": metadata.get("prompt", ""),
                        "image_url": f"/api/images/file/{image_path.relative_to(gallery_path)}",
                        "mood": metadata.get("mood", "contemplative"),
                        "style": metadata.get("style", "digital art"),
                        "reflection": metadata.get(
                            "reflection", metadata.get("prompt", "")
                        ),
                        "created_at": metadata.get("created_at", ""),
                        "thinking": metadata.get("thinking"),
                        "critique_history": metadata.get("critique_history"),
                    }
                )
        except Exception as e:
            logger.debug("failed_to_load_metadata", file=str(json_file), error=str(e))

    # Sort by creation date (newest first)
    portfolio.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return portfolio[:50]  # Limit to 50 most recent


@router.get("/state", response_model=AriaStateResponse)
@limiter.limit("60/minute")
async def get_aria_state(request: Request):
    """Get Aria's current state including mood, energy, personality, and experience."""
    state = _get_aria_state()
    mood_system = state["mood_system"]
    memory = state["memory"]

    # Apply mood decay (emotions fade over time)
    mood_system.apply_decay()

    # Load portfolio from gallery
    portfolio = _load_portfolio_from_gallery()
    state["portfolio"] = portfolio
    state["paintings_created"] = len(portfolio)

    # Get experience progress
    experience_progress = memory.get_experience_progress()

    # Get style axes
    style_axes = (
        mood_system.style_axes.to_dict() if hasattr(mood_system, "style_axes") else None
    )

    return AriaStateResponse(
        name=state["name"],
        mood=mood_system.current_mood.value,
        mood_intensity=getattr(mood_system, "mood_intensity", 0.7),
        energy=mood_system.energy_level,
        feeling=mood_system.describe_feeling(),
        paintings_created=state["paintings_created"],
        personality=state["personality"],
        portfolio=portfolio,
        experience=experience_progress,
        style_axes=style_axes,
    )


@router.post("/create", response_model=AriaCreateResponse)
@limiter.limit("5/minute")
async def create_artwork(request: Request):
    """Trigger Aria to create a new artwork with actual image generation.

    Returns concept info immediately and starts background generation.
    The session_id can be used to track progress via WebSocket.
    """
    from ..core.generator import ImageGenerator
    from ..web.websocket import manager as ws_manager

    state = _get_aria_state()
    mood_system = state["mood_system"]
    profile = state["profile"]
    personality = state["personality"]
    session_id = str(uuid.uuid4())

    try:
        # Get mood influences
        mood = mood_system.current_mood
        mood_influences = mood_system.mood_influences.get(
            mood,
            {
                "styles": ["digital art"],
                "subjects": ["abstract"],
                "colors": ["vibrant colors"],
            },
        )

        # Build concept based on mood
        subject = random.choice(mood_influences.get("subjects", ["abstract"]))
        style = random.choice(mood_influences.get("styles", ["digital art"]))
        colors = random.choice(mood_influences.get("colors", ["vibrant colors"]))

        # Build prompt
        prompt = f"{subject}, {style}, {colors}, masterpiece, highly detailed"
        negative_prompt = "blurry, low quality, distorted, deformed"

        # Generate thinking narrative based on personality
        openness = personality.get("openness", 0.7)
        neuroticism = personality.get("neuroticism", 0.5)

        thinking_parts = []
        if openness > 0.7:
            thinking_parts.append(f"I feel drawn to explore {subject} today.")
        else:
            thinking_parts.append(f"I want to create something about {subject}.")

        if neuroticism > 0.6:
            thinking_parts.append(f"My {mood.value} mood colors everything.")
        else:
            thinking_parts.append(f"I'm feeling {mood.value} and it inspires me.")

        thinking_parts.append(f"I'll use a {style} approach with {colors}.")
        thinking = " ".join(thinking_parts)

        # Simple critique simulation
        critique_history = [
            {
                "critic_name": "Inner Critic",
                "critique": f"The {subject} concept aligns well with your {mood.value} mood.",
                "approved": True,
                "confidence": random.uniform(0.7, 0.95),
            }
        ]

        # Generate reflection
        reflection = profile.reflect_on_creation(
            {"subject": subject, "style": style, "mood": mood.value}
        )

        logger.info(
            "aria_concept_created",
            subject=subject,
            style=style,
            mood=mood.value,
            session_id=session_id,
        )

        # Start background generation
        async def generate_task():
            generator = None
            try:
                config_path = Path("config/config.yaml")
                config = load_config(config_path)
                gallery_path = Path("gallery")

                # Send start event
                await ws_manager.send_generation_start(session_id=session_id)

                # Create generator
                generator = ImageGenerator(
                    model_id=config.model.base_model,
                    device=config.model.device,
                )
                generator.load_model()

                # Progress callback for WebSocket updates
                def on_progress(step: int, total: int, latents: Any = None):
                    asyncio.create_task(
                        ws_manager.send_generation_progress(
                            session_id=session_id,
                            step=step,
                            total_steps=total,
                        )
                    )

                # Generate image
                images = generator.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=768,
                    height=768,
                    num_images=1,
                    on_progress=on_progress,
                )

                # Save image
                if images:
                    import json as json_module

                    now = datetime.now()
                    date_path = gallery_path / now.strftime("%Y/%m/%d") / "archive"
                    date_path.mkdir(parents=True, exist_ok=True)
                    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_noseed.png"
                    save_path = date_path / filename
                    images[0].save(save_path)

                    # Save metadata JSON for gallery API
                    metadata_path = save_path.with_suffix(".json")
                    metadata_path.write_text(
                        json_module.dumps(
                            {
                                "prompt": prompt,
                                "metadata": {
                                    "mood": mood.value,
                                    "subject": subject,
                                    "style": style,
                                    "model": config.model.base_model,
                                },
                                "created_at": now.isoformat(),
                                "featured": False,
                            },
                            indent=2,
                        )
                    )

                    image_url = f"/api/images/file/{now.strftime('%Y/%m/%d')}/archive/{filename}"

                    # Update mood after successful creation
                    mood_system.update_mood()

                    # Send completion event
                    await ws_manager.send_generation_complete(
                        session_id=session_id,
                        image_paths=[image_url],
                        metadata={"prompt": prompt, "mood": mood.value},
                    )

                    logger.info(
                        "aria_image_generated",
                        session_id=session_id,
                        image_url=image_url,
                    )
                else:
                    await ws_manager.send_generation_error(
                        session_id=session_id,
                        error="No valid images generated",
                    )

            except Exception as e:
                logger.error(
                    "aria_generation_failed", error=str(e), session_id=session_id
                )
                await ws_manager.send_generation_error(
                    session_id=session_id,
                    error=str(e),
                )
            finally:
                if generator:
                    generator.clear_vram()

        # Start generation in background
        asyncio.create_task(generate_task())

        return AriaCreateResponse(
            success=True,
            subject=subject,
            style=style,
            prompt=prompt,
            reflection=reflection,
            thinking=thinking,
            critique_history=critique_history,
            session_id=session_id,
        )

    except Exception as e:
        logger.error("aria_create_failed", error=str(e))
        return AriaCreateResponse(success=False, error=str(e))


@router.post("/evolve")
@limiter.limit("10/minute")
async def evolve_state(request: Request):
    """Force Aria's state to evolve."""
    state = _get_aria_state()
    mood_system = state["mood_system"]
    personality = state["personality"]

    # Update mood
    old_mood = mood_system.current_mood
    mood_system.update_mood()
    new_mood = mood_system.current_mood

    # Slowly evolve personality (small changes)
    for trait in personality:
        change = random.uniform(-0.02, 0.02)
        personality[trait] = max(0.0, min(1.0, personality[trait] + change))

    logger.info(
        "aria_evolved",
        old_mood=old_mood.value,
        new_mood=new_mood.value,
        energy=mood_system.energy_level,
    )

    return {
        "mood": new_mood.value,
        "energy": mood_system.energy_level,
        "feeling": mood_system.describe_feeling(),
        "personality": personality,
        "evolved": old_mood != new_mood,
    }


@router.get("/statement", response_model=AriaStatementResponse)
@limiter.limit("30/minute")
async def get_artist_statement(request: Request):
    """Get Aria's artist statement."""
    state = _get_aria_state()
    profile = state["profile"]
    mood_system = state["mood_system"]

    # Get base statement and customize based on current mood
    statement = profile.artist_statement

    # Add mood-influenced postscript
    mood = mood_system.current_mood
    mood_reflections = {
        Mood.CONTEMPLATIVE: "Today I find myself in quiet contemplation, seeking meaning in simplicity.",
        Mood.CHAOTIC: "Right now, I embrace the beautiful chaos within, letting it flow onto the canvas.",
        Mood.MELANCHOLIC: "In this moment, I draw from the deep wells of melancholy, finding beauty in sadness.",
        Mood.ENERGIZED: "I feel alive with creative energy, ready to burst forth with vibrant expression.",
        Mood.REBELLIOUS: "Today I question, I challenge, I rebel against the ordinary.",
        Mood.SERENE: "I exist in a state of peaceful harmony, creating from a place of calm.",
        Mood.RESTLESS: "My spirit is restless, searching for something just beyond reach.",
        Mood.PLAYFUL: "I approach my art with childlike wonder and playful curiosity.",
        Mood.INTROSPECTIVE: "Looking inward, I find infinite landscapes waiting to be explored.",
        Mood.BOLD: "I create with confidence and conviction, making bold statements through my work.",
    }

    full_statement = statement + "\n\n" + mood_reflections.get(mood, "")

    return AriaStatementResponse(statement=full_statement, name=state["name"])


@router.get("/portfolio")
@limiter.limit("30/minute")
async def get_portfolio(request: Request, limit: int = 20):
    """Get Aria's portfolio of creations."""
    portfolio = _load_portfolio_from_gallery()
    return {"count": len(portfolio), "paintings": portfolio[:limit]}


@router.get("/evolution")
@limiter.limit("30/minute")
async def get_evolution(request: Request):
    """Get Aria's artistic evolution timeline.

    Returns:
        Evolution data including:
        - phases: Artistic phases/periods
        - milestones: Notable creations and achievements
        - style_evolution: How style preferences changed over time
        - mood_distribution: Overall mood patterns
        - score_trend: Quality scores over time
    """
    state = _get_aria_state()
    memory = state["memory"]

    # Get evolution data from enhanced memory
    evolution = memory.get_evolution_timeline()
    style_preferences = memory.get_style_preferences_over_time()

    # Add style preferences to evolution data
    evolution["style_preferences"] = style_preferences

    # Add summary statistics
    evolution["summary"] = {
        "total_creations": len(evolution.get("score_trend", [])),
        "unique_styles": len(
            {
                s
                for entry in evolution.get("style_evolution", [])
                for s in entry.get("styles_used", {})
            }
        ),
        "dominant_moods": sorted(
            evolution.get("mood_distribution", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3],
        "phases_count": len(evolution.get("phases", [])),
    }

    logger.info(
        "evolution_data_retrieved",
        phases=len(evolution.get("phases", [])),
        milestones=len(evolution.get("milestones", [])),
    )

    return evolution
