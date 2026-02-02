"""SQLAlchemy database models."""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

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

    # Community gallery fields
    is_public = Column(Boolean, default=False, index=True)
    share_id = Column(String(12), unique=True, index=True)  # Unique share ID
    like_count = Column(Integer, default=0, index=True)
    comment_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0, index=True)

    # Relationships
    likes = relationship(
        "GalleryLike", back_populates="image", cascade="all, delete-orphan"
    )
    comments = relationship(
        "GalleryComment", back_populates="image", cascade="all, delete-orphan"
    )
    shares = relationship(
        "GalleryShare", back_populates="image", cascade="all, delete-orphan"
    )


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


class GalleryLike(Base):  # type: ignore[misc, valid-type]
    """Model for image likes in community gallery."""

    __tablename__ = "gallery_likes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(
        Integer,
        ForeignKey("generated_images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    # Relationship
    image = relationship("GeneratedImage", back_populates="likes")

    __table_args__ = ({"sqlite_autoincrement": True},)


class GalleryComment(Base):  # type: ignore[misc, valid-type]
    """Model for image comments in community gallery."""

    __tablename__ = "gallery_comments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(
        Integer,
        ForeignKey("generated_images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id = Column(String(64), nullable=False, index=True)
    display_name = Column(String(50), default="Anonymous")
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), index=True)

    # Relationship
    image = relationship("GeneratedImage", back_populates="comments")


class GalleryShare(Base):  # type: ignore[misc, valid-type]
    """Model for tracking image shares in community gallery."""

    __tablename__ = "gallery_shares"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(
        Integer,
        ForeignKey("generated_images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    platform = Column(String(20), nullable=False)  # twitter, facebook, etc.
    shared_at = Column(DateTime, default=lambda: datetime.now(UTC))

    # Relationship
    image = relationship("GeneratedImage", back_populates="shares")
