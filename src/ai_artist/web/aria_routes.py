"""Aria API routes - personality, state, and creation endpoints."""

import asyncio
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import torch
from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from ..db.models import GeneratedImage
from ..db.session import get_db
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


class AriaEvolveResponse(BaseModel):
    """Response from evolve endpoint."""

    mood: str
    energy: float
    feeling: str
    personality: dict[str, float]
    evolved: bool


class PortfolioPainting(BaseModel):
    """A single painting in the portfolio."""

    number: int
    subject: str
    prompt: str
    image_url: str
    mood: str
    style: str
    reflection: str
    created_at: str
    thinking: str | None = None
    critique_history: list[dict[str, Any]] | None = None


class AriaPortfolioResponse(BaseModel):
    """Response from portfolio endpoint."""

    count: int
    paintings: list[PortfolioPainting]


class EvolutionSummary(BaseModel):
    """Summary statistics for evolution timeline."""

    total_creations: int
    unique_styles: int
    dominant_moods: list[tuple[str, int]]
    phases_count: int


class AriaEvolutionResponse(BaseModel):
    """Response from evolution endpoint."""

    phases: list[dict[str, Any]] = []
    milestones: list[dict[str, Any]] = []
    style_evolution: list[dict[str, Any]] = []
    mood_distribution: dict[str, int] = {}
    score_trend: list[dict[str, Any]] = []
    style_preferences: list[dict[str, Any]] = []
    summary: EvolutionSummary | None = None


class ReferenceImageUploadResponse(BaseModel):
    """Response from reference image upload."""

    success: bool
    reference_id: str | None = None
    filename: str | None = None
    url: str | None = None
    error: str | None = None


class ReferenceImageListResponse(BaseModel):
    """Response from reference images list."""

    count: int
    references: list[dict[str, Any]]


class CreateWithReferenceRequest(BaseModel):
    """Request for creating artwork with a reference image."""

    reference_id: str
    prompt: str | None = None
    ip_adapter_scale: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Strength of reference image influence (0.0-1.0)",
    )


# Reference images storage path
REFERENCE_IMAGES_PATH = Path("gallery/references")


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


async def _load_portfolio_from_gallery() -> list[dict]:
    """Load portfolio from gallery directory using async file I/O."""
    portfolio: list[dict] = []
    gallery_path = Path("gallery")

    if not gallery_path.exists():
        return portfolio

    # Find all JSON metadata files
    for json_file in gallery_path.rglob("*.json"):
        try:
            async with aiofiles.open(json_file) as f:
                content = await f.read()
                metadata = json.loads(content)

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
    portfolio = await _load_portfolio_from_gallery()
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
async def create_artwork(request: Request, db: Session = Depends(get_db)):
    """Trigger Aria to create a new artwork with actual image generation.

    Returns concept info immediately and starts background generation.
    The session_id can be used to track progress via WebSocket.
    """
    from ..core.generator import ImageGenerator
    from ..inspiration.autonomous import AutonomousInspiration
    from ..web.websocket import manager as ws_manager

    state = _get_aria_state()
    mood_system = state["mood_system"]
    profile = state["profile"]
    personality = state["personality"]
    session_id = str(uuid.uuid4())

    try:
        # Get mood influences for style and colors
        mood = mood_system.current_mood
        mood_influences = mood_system.mood_influences.get(
            mood,
            {
                "styles": ["digital art"],
                "subjects": ["abstract"],
                "colors": ["vibrant colors"],
            },
        )

        # Use comprehensive subject list for variety (not mood-restricted)
        autonomous = AutonomousInspiration()
        subject = random.choice(autonomous.subjects)

        # Use comprehensive style list for variety (mix mood-influenced with autonomous)
        # 70% chance to use autonomous styles, 30% chance to use mood-specific style
        if random.random() < 0.7:
            style = random.choice(autonomous.styles)
        else:
            style = random.choice(mood_influences.get("styles", ["digital art"]))

        # Colors still influenced by mood for emotional coherence
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

        # Send thinking update to WebSocket clients
        await ws_manager.send_thinking_update(
            session_id=session_id, thought_type="observe", content=thinking
        )

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

        # Send reflection as thinking update
        await ws_manager.send_thinking_update(
            session_id=session_id, thought_type="reflect", content=reflection
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
            from ..db.session import get_session_factory

            try:
                config_path = Path("config/config.yaml")
                config = load_config(config_path)
                gallery_path = Path("gallery")

                # Send start event
                await ws_manager.send_generation_start(
                    session_id=session_id, prompt=prompt
                )

                # Send thinking update about starting creation
                await ws_manager.send_thinking_update(
                    session_id=session_id,
                    thought_type="create",
                    content="Beginning the creation process... channeling my vision into form.",
                )

                # Parse dtype correctly
                dtype = (
                    torch.float32 if config.model.dtype == "float32" else torch.float16
                )

                logger.info(
                    "creating_generator",
                    session_id=session_id,
                    device=config.model.device,
                    dtype=config.model.dtype,
                    model=config.model.base_model,
                )

                # Create generator with proper dtype
                generator = ImageGenerator(
                    model_id=config.model.base_model,
                    device=config.model.device,
                    dtype=dtype,
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
                if images and len(images) > 0:
                    import json as json_module

                    now = datetime.now()
                    date_path = gallery_path / now.strftime("%Y/%m/%d") / "archive"
                    date_path.mkdir(parents=True, exist_ok=True)
                    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_noseed.png"
                    save_path = date_path / filename

                    # Save the image file
                    images[0].save(save_path)
                    logger.info("image_saved_to_disk", path=str(save_path))

                    # Save metadata JSON for gallery API
                    metadata_path = save_path.with_suffix(".json")
                    metadata_json = {
                        "prompt": prompt,
                        "metadata": {
                            "mood": mood.value,
                            "subject": subject,
                            "style": style,
                            "model": config.model.base_model,
                        },
                        "created_at": now.isoformat(),
                        "featured": False,
                    }
                    metadata_path.write_text(json_module.dumps(metadata_json, indent=2))

                    image_url = f"/api/images/file/{now.strftime('%Y/%m/%d')}/archive/{filename}"

                    # Save to database
                    try:
                        session_factory = get_session_factory()
                        if session_factory:
                            with session_factory() as db_session:
                                db_image = GeneratedImage(
                                    filename=str(save_path),
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    status=mood.value,
                                    seed=None,
                                    model_id=config.model.base_model,
                                    generation_params={
                                        "width": 768,
                                        "height": 768,
                                        "steps": 30,
                                        "guidance_scale": 7.5,
                                        "subject": subject,
                                        "style": style,
                                    },
                                    final_score=0.8,  # Default score
                                    tags=[mood.value, subject, style],
                                    created_at=now,
                                )
                                db_session.add(db_image)
                                db_session.commit()
                                logger.info("image_saved_to_database", id=db_image.id)
                        else:
                            logger.warning("database_not_configured_skipping_db_save")
                    except Exception as db_error:
                        logger.error("database_save_failed", error=str(db_error))
                        # Don't fail the whole operation if DB save fails

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
                    error_msg = "No valid images generated. This often happens with MPS + float16. Try using float32 in config.yaml."
                    logger.error(
                        "generation_produced_no_images",
                        session_id=session_id,
                        device=config.model.device,
                        dtype=config.model.dtype,
                        hint="Set dtype: 'float32' in config/config.yaml if using MPS",
                    )
                    await ws_manager.send_generation_error(
                        session_id=session_id,
                        error=error_msg,
                    )

            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                logger.error(
                    "aria_generation_failed",
                    error=str(e),
                    session_id=session_id,
                    traceback=error_details,
                )
                await ws_manager.send_generation_error(
                    session_id=session_id,
                    error=f"Generation failed: {str(e)}",
                )
            finally:
                if generator:
                    generator.clear_vram()

        # Start generation in background with exception handling
        task = asyncio.create_task(generate_task())

        # Add exception handler for the background task
        def handle_task_exception(task):
            try:
                task.result()
            except Exception as e:
                logger.error(
                    "background_task_exception", error=str(e), session_id=session_id
                )

        task.add_done_callback(handle_task_exception)

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


@router.post("/evolve", response_model=AriaEvolveResponse)
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

    return AriaEvolveResponse(
        mood=new_mood.value,
        energy=mood_system.energy_level,
        feeling=mood_system.describe_feeling(),
        personality=personality,
        evolved=old_mood != new_mood,
    )


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


@router.get("/portfolio", response_model=AriaPortfolioResponse)
@limiter.limit("30/minute")
async def get_portfolio(request: Request, limit: int = 20):
    """Get Aria's portfolio of creations."""
    portfolio_dicts = await _load_portfolio_from_gallery()
    paintings = [PortfolioPainting.model_validate(p) for p in portfolio_dicts[:limit]]
    return AriaPortfolioResponse(count=len(portfolio_dicts), paintings=paintings)


@router.get("/evolution", response_model=AriaEvolutionResponse)
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

    # Build summary statistics
    summary = EvolutionSummary(
        total_creations=len(evolution.get("score_trend", [])),
        unique_styles=len(
            {
                s
                for entry in evolution.get("style_evolution", [])
                for s in entry.get("styles_used", {})
            }
        ),
        dominant_moods=sorted(
            evolution.get("mood_distribution", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3],
        phases_count=len(evolution.get("phases", [])),
    )

    logger.info(
        "evolution_data_retrieved",
        phases=len(evolution.get("phases", [])),
        milestones=len(evolution.get("milestones", [])),
    )

    return AriaEvolutionResponse(
        phases=evolution.get("phases", []),
        milestones=evolution.get("milestones", []),
        style_evolution=evolution.get("style_evolution", []),
        mood_distribution=evolution.get("mood_distribution", {}),
        score_trend=evolution.get("score_trend", []),
        style_preferences=style_preferences,
        summary=summary,
    )


# =============================================================================
# Async Job Queue Endpoints
# =============================================================================


class AsyncGenerationRequest(BaseModel):
    """Request model for async generation."""

    prompt: str | None = None  # Optional - will use mood-based prompt if not provided
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: int | None = None
    priority: str = "normal"  # high, normal, low


class AsyncGenerationResponse(BaseModel):
    """Response from async generation endpoint."""

    success: bool
    job_id: str | None = None
    message: str
    queue_position: int | None = None
    estimated_wait_seconds: int | None = None


class JobStatusResponse(BaseModel):
    """Job status response."""

    job_id: str
    status: str
    progress: int = 0
    result: dict | None = None
    enqueued_at: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    error: str | None = None


class QueueStatsResponse(BaseModel):
    """Queue statistics response."""

    enabled: bool
    queues: dict[str, dict] = {}


@router.post("/generate-async", response_model=AsyncGenerationResponse)
@limiter.limit("10/minute")
async def generate_async(
    request: Request,
    generation_request: AsyncGenerationRequest,
):
    """Start an async image generation job.

    This endpoint enqueues a generation job and returns immediately with a job_id.
    Use GET /aria/job/{job_id} to check status, or connect to WebSocket for
    real-time progress updates.

    Priority options:
    - high: Processed first, for premium/urgent requests
    - normal: Standard priority (default)
    - low: Background jobs, processed when queue is empty

    Returns:
        AsyncGenerationResponse with job_id for tracking
    """

    from ..queue import get_queue
    from .websocket import manager as ws_manager

    # Get or initialize queue
    queue = get_queue()

    if not queue.is_available():
        return AsyncGenerationResponse(
            success=False,
            message="Job queue is not available. Please use sync /create endpoint instead.",
        )

    state = _get_aria_state()
    mood_system = state["mood_system"]

    # Generate prompt if not provided (mood-based)
    prompt = generation_request.prompt
    if not prompt:
        from ..inspiration.autonomous import AutonomousInspiration

        mood = mood_system.current_mood
        mood_influences = mood_system.mood_influences.get(
            mood,
            {
                "styles": ["digital art"],
                "subjects": ["abstract"],
                "colors": ["vibrant colors"],
            },
        )

        autonomous = AutonomousInspiration()
        subject = random.choice(autonomous.subjects)

        if random.random() < 0.7:
            style = random.choice(autonomous.styles)
        else:
            style = random.choice(mood_influences.get("styles", ["digital art"]))

        colors = random.choice(mood_influences.get("colors", ["vibrant colors"]))
        prompt = f"{subject}, {style}, {colors}, masterpiece, highly detailed"

    # Prepare generation parameters
    params = {
        "width": generation_request.width,
        "height": generation_request.height,
        "num_inference_steps": generation_request.num_inference_steps,
        "guidance_scale": generation_request.guidance_scale,
        "num_images": generation_request.num_images,
        "seed": generation_request.seed,
        "negative_prompt": generation_request.negative_prompt
        or "blurry, low quality, distorted, deformed",
    }

    # Enqueue job
    job_id = queue.enqueue_generation(
        prompt=prompt,
        params=params,
        priority=generation_request.priority,
        meta={
            "mood": mood_system.current_mood.value,
            "source": "aria_api",
        },
    )

    if not job_id:
        return AsyncGenerationResponse(
            success=False,
            message="Failed to enqueue job. Please try again.",
        )

    # Get queue stats for position estimate
    stats = queue.get_queue_stats()
    queue_name = (
        f"generation-{generation_request.priority}"
        if generation_request.priority != "normal"
        else "generation"
    )
    queue_info = stats.get("queues", {}).get(generation_request.priority, {})
    queue_count = queue_info.get("count", 0)

    # Estimate wait time (rough: 60 seconds per job)
    estimated_wait = queue_count * 60

    # Notify WebSocket clients about new job
    await ws_manager.broadcast(
        {
            "type": "job_enqueued",
            "job_id": job_id,
            "prompt": prompt[:100],
            "priority": generation_request.priority,
            "timestamp": datetime.now().isoformat(),
        }
    )

    logger.info(
        "async_generation_enqueued",
        job_id=job_id,
        priority=generation_request.priority,
        queue_position=queue_count,
    )

    return AsyncGenerationResponse(
        success=True,
        job_id=job_id,
        message="Generation job enqueued successfully",
        queue_position=queue_count,
        estimated_wait_seconds=estimated_wait,
    )


@router.get("/job/{job_id}", response_model=JobStatusResponse)
@limiter.limit("60/minute")
async def get_job_status(request: Request, job_id: str):
    """Get the status of a generation job.

    Returns current status, progress percentage, and results if complete.

    Status values:
    - queued: Waiting in queue
    - started: Currently processing
    - finished: Complete - check result field
    - failed: Failed - check error field
    """
    from fastapi import HTTPException

    from ..queue import get_queue

    queue = get_queue()

    if not queue.is_available():
        raise HTTPException(status_code=503, detail="Job queue is not available")

    job_info = queue.get_job_status(job_id)

    if not job_info:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job_info.id,
        status=job_info.status.value,
        progress=job_info.progress,
        result=job_info.result if isinstance(job_info.result, dict) else None,
        enqueued_at=job_info.enqueued_at,
        started_at=job_info.started_at,
        ended_at=job_info.ended_at,
        error=job_info.error,
    )


@router.delete("/job/{job_id}")
@limiter.limit("30/minute")
async def cancel_job(request: Request, job_id: str):
    """Cancel a queued job.

    Only queued jobs can be cancelled. Jobs that are already started
    will continue to completion.
    """
    from fastapi import HTTPException

    from ..queue import get_queue

    queue = get_queue()

    if not queue.is_available():
        raise HTTPException(status_code=503, detail="Job queue is not available")

    # Check if job exists and is cancellable
    job_info = queue.get_job_status(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job_info.status.value not in ("queued", "deferred", "scheduled"):
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be cancelled - status is {job_info.status.value}",
        )

    success = queue.cancel_job(job_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel job")

    logger.info("job_cancelled", job_id=job_id)

    return {"success": True, "message": f"Job {job_id} cancelled"}


@router.get("/queue/stats", response_model=QueueStatsResponse)
@limiter.limit("60/minute")
async def get_queue_stats(request: Request):
    """Get queue statistics.

    Returns information about all priority queues including:
    - Number of jobs waiting
    - Number of jobs in progress
    - Number of completed/failed jobs
    """
    from ..queue import get_queue

    queue = get_queue()
    stats = queue.get_queue_stats()

    return QueueStatsResponse(
        enabled=stats.get("enabled", False),
        queues=stats.get("queues", {}),
    )


# =============================================================================
# Reference Image Endpoints (IP-Adapter Support)
# =============================================================================

import io


@router.post("/reference-image", response_model=ReferenceImageUploadResponse)
@limiter.limit("10/minute")
async def upload_reference_image(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload a reference image for IP-Adapter style transfer.

    The reference image will be used to guide the style/composition of
    future artwork generations.

    Args:
        file: Image file (PNG, JPEG, WebP supported)

    Returns:
        Reference ID that can be used in create requests
    """
    try:
        # Validate file type
        allowed_types = {"image/png", "image/jpeg", "image/webp", "image/jpg"}
        if file.content_type not in allowed_types:
            return ReferenceImageUploadResponse(
                success=False,
                error=f"Invalid file type: {file.content_type}. Allowed: PNG, JPEG, WebP",
            )

        # Create references directory if needed
        REFERENCE_IMAGES_PATH.mkdir(parents=True, exist_ok=True)

        # Generate unique ID
        reference_id = str(uuid.uuid4())[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine extension from content type
        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
        }
        ext = ext_map.get(file.content_type, ".png")
        filename = f"ref_{timestamp}_{reference_id}{ext}"
        save_path = REFERENCE_IMAGES_PATH / filename

        # Read and validate image
        contents = await file.read()

        # Check file size limit (10 MB max)
        MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
        if len(contents) > MAX_UPLOAD_SIZE:
            return ReferenceImageUploadResponse(
                success=False,
                error=f"File too large ({len(contents) // (1024 * 1024)}MB). Maximum size is 10MB",
            )

        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()  # Verify it's a valid image
            # Re-open after verify (verify closes the file)
            img = Image.open(io.BytesIO(contents))
        except Exception as e:
            return ReferenceImageUploadResponse(
                success=False,
                error=f"Invalid image file: {str(e)}",
            )

        # Convert to RGB if needed (for consistency)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # Save the image
        img.save(save_path)

        # Save metadata
        metadata = {
            "reference_id": reference_id,
            "filename": filename,
            "original_name": file.filename,
            "content_type": file.content_type,
            "width": img.width,
            "height": img.height,
            "uploaded_at": datetime.now().isoformat(),
        }
        metadata_path = save_path.with_suffix(".json")
        async with aiofiles.open(metadata_path, "w") as f:
            await f.write(json.dumps(metadata, indent=2))

        logger.info(
            "reference_image_uploaded",
            reference_id=reference_id,
            filename=filename,
            size=(img.width, img.height),
        )

        return ReferenceImageUploadResponse(
            success=True,
            reference_id=reference_id,
            filename=filename,
            url=f"/api/images/file/references/{filename}",
        )

    except Exception as e:
        logger.error("reference_image_upload_failed", error=str(e))
        return ReferenceImageUploadResponse(
            success=False,
            error=f"Upload failed: {str(e)}",
        )


@router.get("/reference-images", response_model=ReferenceImageListResponse)
@limiter.limit("30/minute")
async def list_reference_images(request: Request, limit: int = 20):
    """List uploaded reference images.

    Returns:
        List of reference images with their metadata
    """
    references = []

    if REFERENCE_IMAGES_PATH.exists():
        for json_file in REFERENCE_IMAGES_PATH.glob("*.json"):
            try:
                async with aiofiles.open(json_file) as f:
                    content = await f.read()
                    metadata = json.loads(content)

                # Add URL for display
                metadata["url"] = f"/api/images/file/references/{metadata['filename']}"
                references.append(metadata)
            except Exception as e:
                logger.debug(
                    "failed_to_load_reference_metadata",
                    file=str(json_file),
                    error=str(e),
                )

    # Sort by upload date (newest first)
    references.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)

    return ReferenceImageListResponse(
        count=len(references),
        references=references[:limit],
    )


@router.delete("/reference-image/{reference_id}")
@limiter.limit("10/minute")
async def delete_reference_image(request: Request, reference_id: str):
    """Delete a reference image.

    Args:
        reference_id: The reference ID to delete

    Returns:
        Success status
    """
    if not REFERENCE_IMAGES_PATH.exists():
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Reference not found"},
        )

    # Find the reference by ID
    for json_file in REFERENCE_IMAGES_PATH.glob("*.json"):
        try:
            async with aiofiles.open(json_file) as f:
                content = await f.read()
                metadata = json.loads(content)

            if metadata.get("reference_id") == reference_id:
                # Delete image and metadata
                image_path = REFERENCE_IMAGES_PATH / metadata["filename"]
                if image_path.exists():
                    image_path.unlink()
                json_file.unlink()

                logger.info("reference_image_deleted", reference_id=reference_id)
                return {"success": True, "reference_id": reference_id}
        except Exception:
            continue

    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Reference not found"},
    )


@router.post("/create-with-reference", response_model=AriaCreateResponse)
@limiter.limit("5/minute")
async def create_with_reference(
    request: Request,
    body: CreateWithReferenceRequest,
    db: Session = Depends(get_db),
):
    """Create artwork using a reference image for style transfer.

    Uses IP-Adapter to condition generation on the reference image's
    visual characteristics (style, composition, colors).

    Args:
        body: Request with reference_id, optional prompt, and ip_adapter_scale

    Returns:
        Same as /create endpoint, but with reference-conditioned generation
    """
    from ..core.generator import ImageGenerator
    from ..inspiration.autonomous import AutonomousInspiration
    from ..web.websocket import manager as ws_manager

    state = _get_aria_state()
    mood_system = state["mood_system"]
    profile = state["profile"]
    personality = state["personality"]
    session_id = str(uuid.uuid4())

    # Find reference image
    reference_path = None
    if REFERENCE_IMAGES_PATH.exists():
        for json_file in REFERENCE_IMAGES_PATH.glob("*.json"):
            try:
                async with aiofiles.open(json_file) as f:
                    content = await f.read()
                    metadata = json.loads(content)

                if metadata.get("reference_id") == body.reference_id:
                    reference_path = REFERENCE_IMAGES_PATH / metadata["filename"]
                    break
            except Exception:
                continue

    if reference_path is None or not reference_path.exists():
        return AriaCreateResponse(
            success=False,
            error=f"Reference image not found: {body.reference_id}",
        )

    try:
        mood = mood_system.current_mood
        mood_influences = mood_system.mood_influences.get(
            mood,
            {
                "styles": ["digital art"],
                "subjects": ["abstract"],
                "colors": ["vibrant colors"],
            },
        )

        # Use provided prompt or generate one
        if body.prompt:
            prompt = body.prompt
            subject = prompt.split(",")[0].strip()[:50]
            style = "reference-guided"
        else:
            autonomous = AutonomousInspiration()
            subject = random.choice(autonomous.subjects)
            style = random.choice(autonomous.styles)
            colors = random.choice(mood_influences.get("colors", ["vibrant colors"]))
            prompt = f"{subject}, {style}, {colors}, masterpiece, highly detailed"

        negative_prompt = "blurry, low quality, distorted, deformed"

        # Generate thinking narrative
        openness = personality.get("openness", 0.7)
        thinking_parts = [
            "I'm drawing inspiration from a reference image today.",
            f"Using it as a guide for {subject}.",
        ]
        if openness > 0.7:
            thinking_parts.append(
                f"The reference's essence will blend with my {mood.value} mood."
            )
        thinking = " ".join(thinking_parts)

        await ws_manager.send_thinking_update(
            session_id=session_id, thought_type="observe", content=thinking
        )

        critique_history = [
            {
                "critic_name": "Style Advisor",
                "critique": f"Reference image will guide the {style} approach effectively.",
                "approved": True,
                "confidence": random.uniform(0.75, 0.95),
            }
        ]

        reflection = profile.reflect_on_creation(
            {"subject": subject, "style": style, "mood": mood.value, "reference": True}
        )

        await ws_manager.send_thinking_update(
            session_id=session_id, thought_type="reflect", content=reflection
        )

        logger.info(
            "aria_reference_concept_created",
            subject=subject,
            style=style,
            reference_id=body.reference_id,
            ip_adapter_scale=body.ip_adapter_scale,
            session_id=session_id,
        )

        # Background generation with reference
        async def generate_task():
            generator = None
            from ..db.session import get_session_factory

            try:
                config_path = Path("config/config.yaml")
                config = load_config(config_path)
                gallery_path = Path("gallery")

                await ws_manager.send_generation_start(
                    session_id=session_id, prompt=prompt
                )

                await ws_manager.send_thinking_update(
                    session_id=session_id,
                    thought_type="create",
                    content="Blending reference style with my vision...",
                )

                dtype = (
                    torch.float32 if config.model.dtype == "float32" else torch.float16
                )

                # Load reference image
                ref_image = Image.open(reference_path)

                generator = ImageGenerator(
                    model_id=config.model.base_model,
                    device=config.model.device,
                    dtype=dtype,
                )
                generator.load_model()

                def on_progress(step: int, total: int, latents=None):
                    asyncio.create_task(
                        ws_manager.send_generation_progress(
                            session_id=session_id,
                            step=step,
                            total_steps=total,
                        )
                    )

                # Generate with reference image
                images = generator.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=768,
                    height=768,
                    num_images=1,
                    on_progress=on_progress,
                    reference_image=ref_image,
                    ip_adapter_scale=body.ip_adapter_scale,
                )

                if images and len(images) > 0:
                    import json as json_module

                    now = datetime.now()
                    date_path = gallery_path / now.strftime("%Y/%m/%d") / "archive"
                    date_path.mkdir(parents=True, exist_ok=True)
                    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_ref.png"
                    save_path = date_path / filename

                    images[0].save(save_path)
                    logger.info("reference_image_generated", path=str(save_path))

                    metadata_path = save_path.with_suffix(".json")
                    metadata_json = {
                        "prompt": prompt,
                        "metadata": {
                            "mood": mood.value,
                            "subject": subject,
                            "style": style,
                            "model": config.model.base_model,
                            "reference_id": body.reference_id,
                            "ip_adapter_scale": body.ip_adapter_scale,
                        },
                        "created_at": now.isoformat(),
                        "featured": False,
                    }
                    metadata_path.write_text(json_module.dumps(metadata_json, indent=2))

                    image_url = f"/api/images/file/{now.strftime('%Y/%m/%d')}/archive/{filename}"

                    try:
                        session_factory = get_session_factory()
                        if session_factory:
                            with session_factory() as db_session:
                                db_image = GeneratedImage(
                                    filename=str(save_path),
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    status=mood.value,
                                    seed=None,
                                    model_id=config.model.base_model,
                                    generation_params={
                                        "width": 768,
                                        "height": 768,
                                        "steps": 30,
                                        "guidance_scale": 7.5,
                                        "subject": subject,
                                        "style": style,
                                        "reference_id": body.reference_id,
                                        "ip_adapter_scale": body.ip_adapter_scale,
                                    },
                                    final_score=0.85,
                                    tags=[mood.value, subject, style, "reference"],
                                    created_at=now,
                                )
                                db_session.add(db_image)
                                db_session.commit()
                    except Exception as db_error:
                        logger.error("database_save_failed", error=str(db_error))

                    mood_system.update_mood()

                    await ws_manager.send_generation_complete(
                        session_id=session_id,
                        image_paths=[image_url],
                        metadata={
                            "prompt": prompt,
                            "mood": mood.value,
                            "reference_id": body.reference_id,
                        },
                    )
                else:
                    error_msg = "No valid images generated with reference."
                    logger.error("reference_generation_failed", session_id=session_id)
                    await ws_manager.send_generation_error(
                        session_id=session_id, error=error_msg
                    )

            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                logger.error(
                    "aria_reference_generation_failed",
                    error=str(e),
                    session_id=session_id,
                    traceback=error_details,
                )
                await ws_manager.send_generation_error(
                    session_id=session_id,
                    error=f"Generation failed: {str(e)}",
                )
            finally:
                if generator:
                    generator.clear_vram()

        task = asyncio.create_task(generate_task())

        def handle_task_exception(task):
            try:
                task.result()
            except Exception as e:
                logger.error(
                    "background_task_exception", error=str(e), session_id=session_id
                )

        task.add_done_callback(handle_task_exception)

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
        logger.error("aria_create_with_reference_failed", error=str(e))
        return AriaCreateResponse(success=False, error=str(e))
