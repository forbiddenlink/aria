"""SQLAlchemy database models."""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class GeneratedImage(Base):  # type: ignore[misc, valid-type]
    """Model for generated artwork."""

    __tablename__ = "generated_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False, unique=True, index=True)
    prompt = Column(String, nullable=False)
    negative_prompt = Column(String, default="")

    # Source information
    source_url = Column(String)
    source_query = Column(String)

    # Generation parameters
    model_id = Column(String, nullable=False, index=True)
    generation_params = Column(JSON, default=dict)
    seed = Column(Integer)

    # Quality metrics
    aesthetic_score = Column(Float)
    clip_score = Column(Float)
    technical_score = Column(Float)
    final_score = Column(Float, index=True)

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), index=True)
    status = Column(String, default="pending", index=True)  # pending, curated, rejected
    is_featured = Column(Boolean, default=False)
    tags = Column(JSON, default=list)


class TrainingSession(Base):  # type: ignore[misc, valid-type]
    """Model for LoRA training sessions."""

    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    model_path = Column(String, nullable=False)

    # Training configuration
    config = Column(JSON, default=dict)
    dataset_size = Column(Integer)

    # Training metrics
    final_loss = Column(Float)
    training_time_seconds = Column(Float)
    metrics = Column(JSON, default=dict)

    # Timestamps
    started_at = Column(DateTime, default=lambda: datetime.now(UTC))
    completed_at = Column(DateTime)

    status = Column(String, default="running")  # running, completed, failed


class CreationSession(Base):  # type: ignore[misc, valid-type]
    """Model for automated creation sessions."""

    __tablename__ = "creation_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    theme = Column(String)
    images_created = Column(Integer, default=0)
    images_kept = Column(Integer, default=0)
    avg_score = Column(Float)
    started_at = Column(DateTime, default=lambda: datetime.now(UTC))
    completed_at = Column(DateTime)
