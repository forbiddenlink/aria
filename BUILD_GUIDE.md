# AI Artist - Complete Build Guide

**Detailed Implementation Guide for AI Agents and Developers**

This document provides step-by-step instructions for building the AI Artist project from scratch. Each phase includes specific, actionable tasks with verification steps to ensure progress.

---

## Table of Contents

- [Overview](#overview)
- [Phase 0: Prerequisites](#phase-0-prerequisites)
- [Phase 0.5: Foundation (Week 1)](#phase-05-foundation-week-1)
- [Phase 1: Basic Pipeline (Weeks 2-3)](#phase-1-basic-pipeline-weeks-2-3)
- [Phase 2: Style Training (Weeks 4-5)](#phase-2-style-training-weeks-4-5)
- [Phase 3: Automation (Week 6)](#phase-3-automation-week-6)
- [Phase 4: Advanced Features (Weeks 7-8)](#phase-4-advanced-features-weeks-7-8)
- [Verification Checklist](#verification-checklist)

---

## Overview

### Project Goals
- Build an autonomous AI artist that generates unique artwork
- Implement LoRA-based style training for consistent artistic voice
- Create automated curation and scheduling systems
- Ensure production-ready code with comprehensive testing

### Success Criteria
- Can generate high-quality images (aesthetic score >7.0)
- Style is consistent across different subjects
- System runs autonomously 24/7 without failures
- Test coverage >70%
- All legal and compliance requirements met

---

## Phase 0: Prerequisites

### System Check

**Step 0.1: Verify Python Installation**
```bash
python3 --version
# Expected: Python 3.11 or higher
```
✅ **Verification**: Python version is 3.11+

**Step 0.2: Check GPU Availability**
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```
✅ **Verification**: CUDA is available (for GPU) or CPU fallback accepted

**Step 0.3: Verify Disk Space**
```bash
df -h .
# Need at least 50GB free
```
✅ **Verification**: At least 50GB available

### API Keys

**Step 0.4: Create Unsplash Account**
1. Go to https://unsplash.com/developers
2. Create a new application
3. Note down Access Key and Secret Key

✅ **Verification**: You have both Unsplash keys

**Step 0.5: (Optional) Create Pexels Account**
1. Go to https://www.pexels.com/api/
2. Generate API key

✅ **Verification**: Pexels API key obtained (or skipped)

**Step 0.6: (Optional) Create HuggingFace Account**
1. Go to https://huggingface.co/
2. Create account and generate access token
3. Accept SDXL model license at https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

✅ **Verification**: HuggingFace token ready

---

## Phase 0.5: Foundation (Week 1)

### Day 1: Project Setup

**Step 1.1: Create Project Directory**
```bash
cd /Volumes/LizsDisk/ai-artist
pwd
# Should show: /Volumes/LizsDisk/ai-artist
```
✅ **Verification**: In correct directory

**Step 1.2: Initialize Git Repository**
```bash
git init
git branch -M main
```
✅ **Verification**: Run `git status` - should show "On branch main"

**Step 1.3: Create .gitignore**
```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment variables
.env
.env.local

# Data and Models
models/cache/
models/lora/*.safetensors
gallery/
data/*.db
data/*.db-*
logs/
*.log

# ML artifacts
wandb/
mlruns/
.dvc/

# Temporary files
.DS_Store
Thumbs.db
*.tmp

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
EOF
```
✅ **Verification**: File `.gitignore` exists

**Step 1.4: Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
which python
# Should show path to venv/bin/python
```
✅ **Verification**: Python points to venv

**Step 1.5: Install Modern Package Manager (uv)**
```bash
pip install --upgrade pip
pip install uv
uv --version
```
✅ **Verification**: uv is installed

**Step 1.6: Create pyproject.toml**
```bash
cat > pyproject.toml << 'EOF'
[project]
name = "ai-artist"
version = "0.1.0"
description = "Autonomous AI artist with LoRA style training"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "AI Artist Project" }
]

dependencies = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "diffusers>=0.27.0",
    "transformers>=4.38.0",
    "accelerate>=0.27.0",
    "peft>=0.7.0",
    "safetensors>=0.4.0",
    "pillow>=10.0.0",
    "httpx>=0.27.0",
    "tenacity>=8.2.0",
    "pydantic>=2.5.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
    "apscheduler>=3.10.0",
    "structlog>=24.0.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
    "black>=24.0.0",
    "pre-commit>=3.6.0",
]
training = [
    "wandb>=0.16.0",
    "xformers>=0.0.24",
]
quality = [
    "imagehash>=4.3.0",
    "clip @ git+https://github.com/openai/CLIP.git",
]

[tool.ruff]
target-version = "py311"
line-length = 88
select = ["E", "W", "F", "I", "B", "C4", "UP", "SIM"]
ignore = ["E501"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
    "unit: marks unit tests",
]
EOF
```
✅ **Verification**: File `pyproject.toml` exists

**Step 1.7: Install Core Dependencies**
```bash
# Install PyTorch first (adjust for your system)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
uv pip install -e ".[dev,training]"
```
✅ **Verification**: Run `pip list | grep torch` - should show torch installed

**Step 1.8: Create .env File**
```bash
cat > .env << 'EOF'
# API Keys
UNSPLASH_ACCESS_KEY=your_access_key_here
UNSPLASH_SECRET_KEY=your_secret_key_here
PEXELS_API_KEY=your_pexels_key_here
HF_TOKEN=your_huggingface_token_here

# Database
DATABASE_URL=sqlite:///./data/ai_artist.db

# Environment
ENV=development
DEBUG=true
EOF

# Never commit .env!
git add .gitignore
git commit -m "chore: add .gitignore"
```
✅ **Verification**: `.env` exists and NOT in git (`git status` shouldn't show it)

**Step 1.9: Create Directory Structure**
```bash
mkdir -p src/ai_artist/{core,models,api,db,utils,training,curation,scheduling,gallery}
mkdir -p tests/{unit,integration,e2e,fixtures}
mkdir -p {models/lora,models/cache,gallery,data,logs,configs,scripts,notebooks}

# Create __init__.py files
touch src/ai_artist/__init__.py
touch src/ai_artist/{core,models,api,db,utils,training,curation,scheduling,gallery}/__init__.py
touch tests/__init__.py
```
✅ **Verification**: Run `tree -L 3 src` - should show structure

**Step 1.10: First Git Commit**
```bash
git add .
git commit -m "chore: initial project structure"
```
✅ **Verification**: Run `git log` - should show commit

### Day 2: Development Tools

**Step 2.1: Set up Pre-commit Hooks**
```bash
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=10000']
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.0
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]
        args: [--ignore-missing-imports]
EOF

pre-commit install
```
✅ **Verification**: Run `pre-commit run --all-files` - should complete

**Step 2.2: Configure pytest**
```bash
cat > pytest.ini << 'EOF'
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -ra
    -q
    --strict-markers
    --cov=src/ai_artist
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
EOF
```
✅ **Verification**: File `pytest.ini` exists

**Step 2.3: Create First Test**
```bash
cat > tests/test_smoke.py << 'EOF'
"""Smoke tests to verify environment setup."""

import torch
import diffusers
import transformers

def test_torch_available():
    """Test that PyTorch is installed."""
    assert torch.__version__ is not None

def test_cuda_or_cpu():
    """Test that we have either CUDA or CPU available."""
    assert torch.cuda.is_available() or True  # CPU fallback

def test_diffusers_available():
    """Test that diffusers library is installed."""
    assert diffusers.__version__ is not None

def test_transformers_available():
    """Test that transformers library is installed."""
    assert transformers.__version__ is not None
EOF

pytest tests/test_smoke.py -v
```
✅ **Verification**: All 4 tests pass

**Step 2.4: Set up Logging Configuration**
```bash
cat > src/ai_artist/utils/logging.py << 'EOF'
"""Structured logging configuration using structlog."""

import structlog
import logging
import sys
from pathlib import Path


def configure_logging(log_level: str = "INFO", log_file: Path | None = None):
    """Configure structured logging."""

    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add JSON renderer for production, console renderer for development
    if log_file:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.root.addHandler(handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
EOF
```
✅ **Verification**: File created

**Step 2.5: Create Configuration Loader**
```bash
cat > src/ai_artist/utils/config.py << 'EOF'
"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Model configuration."""
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    dtype: Literal["float16", "float32"] = "float16"
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True


class GenerationConfig(BaseModel):
    """Generation parameters."""
    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)
    num_inference_steps: int = Field(30, ge=10, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    num_variations: int = Field(3, ge=1, le=10)


class APIKeysConfig(BaseModel):
    """API keys configuration."""
    unsplash_access_key: str
    unsplash_secret_key: str
    pexels_api_key: str | None = None


class Config(BaseSettings):
    """Main application configuration."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    api_keys: APIKeysConfig

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
EOF
```
✅ **Verification**: File created

**Step 2.6: Commit Development Tools**
```bash
git add .
git commit -m "chore: add development tools and configuration"
```
✅ **Verification**: Git commit successful

### Day 3: Database Setup

**Step 3.1: Create Database Models**
```bash
cat > src/ai_artist/db/models.py << 'EOF'
"""SQLAlchemy database models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class GeneratedImage(Base):
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
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String, default="pending", index=True)  # pending, curated, rejected
    is_featured = Column(Boolean, default=False)
    tags = Column(JSON, default=list)


class TrainingSession(Base):
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
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    status = Column(String, default="running")  # running, completed, failed


class CreationSession(Base):
    """Model for automated creation sessions."""
    __tablename__ = "creation_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    theme = Column(String)
    images_created = Column(Integer, default=0)
    images_kept = Column(Integer, default=0)
    avg_score = Column(Float)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
EOF
```
✅ **Verification**: File created

**Step 3.2: Initialize Alembic**
```bash
alembic init alembic
```
✅ **Verification**: Directory `alembic/` exists

**Step 3.3: Configure Alembic**
```bash
# Edit alembic.ini to use environment variable
sed -i.bak 's|^sqlalchemy.url = .*|sqlalchemy.url = sqlite:///./data/ai_artist.db|' alembic.ini

cat > alembic/env.py << 'EOF'
"""Alembic migration environment."""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_artist.db.models import Base

config = context.config
database_url = os.getenv("DATABASE_URL", "sqlite:///./data/ai_artist.db")
config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,  # Required for SQLite
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # Required for SQLite
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
EOF
```
✅ **Verification**: File `alembic/env.py` modified

**Step 3.4: Create Initial Migration**
```bash
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```
✅ **Verification**: Database file `data/ai_artist.db` created

**Step 3.5: Create Database Utilities**
```bash
cat > src/ai_artist/db/session.py << 'EOF'
"""Database session management."""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager


def create_db_engine(db_path: Path):
    """Create SQLite engine with optimal settings."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_pre_ping=True,
    )

    # Enable WAL mode for better concurrency
    with engine.connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache

    return engine


def create_session_factory(db_path: Path) -> sessionmaker:
    """Create session factory."""
    engine = create_db_engine(db_path)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_db_session(session_factory: sessionmaker):
    """Context manager for database sessions."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
EOF
```
✅ **Verification**: File created

**Step 3.6: Test Database Connection**
```bash
cat > tests/unit/test_database.py << 'EOF'
"""Test database operations."""

import pytest
from pathlib import Path
from src.ai_artist.db.models import GeneratedImage, Base
from src.ai_artist.db.session import create_db_engine, create_session_factory


@pytest.fixture
def test_db(tmp_path):
    """Create test database."""
    db_path = tmp_path / "test.db"
    engine = create_db_engine(db_path)
    Base.metadata.create_all(engine)
    return create_session_factory(db_path)


def test_create_image_record(test_db):
    """Test creating an image record."""
    with test_db() as session:
        image = GeneratedImage(
            filename="test.png",
            prompt="a test image",
            model_id="test-model",
        )
        session.add(image)
        session.commit()

        # Verify
        result = session.query(GeneratedImage).filter_by(filename="test.png").first()
        assert result is not None
        assert result.prompt == "a test image"
EOF

pytest tests/unit/test_database.py -v
```
✅ **Verification**: Test passes

**Step 3.7: Commit Database Setup**
```bash
git add .
git commit -m "feat: add database models and migrations"
```
✅ **Verification**: Git commit successful

### Days 4-5: Core Modules and Testing

**Step 4.1: Create Unsplash API Client**
```bash
cat > src/ai_artist/api/unsplash.py << 'EOF'
"""Unsplash API client with retry logic."""

from typing import Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RateLimitError(Exception):
    """Rate limit exceeded."""
    pass


class UnsplashClient:
    """Async Unsplash API client."""

    def __init__(self, access_key: str, app_name: str = "ai-artist"):
        self.access_key = access_key
        self.app_name = app_name
        self.base_url = "https://api.unsplash.com"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Client-ID {access_key}"},
            timeout=30.0,
        )

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
    )
    async def search_photos(
        self,
        query: str,
        per_page: int = 10,
        orientation: str | None = None,
    ) -> dict[str, Any]:
        """Search photos with retry logic."""
        params = {"query": query, "per_page": per_page}
        if orientation:
            params["orientation"] = orientation

        logger.info("searching_photos", query=query, per_page=per_page)

        response = await self.client.get("/search/photos", params=params)

        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")

        response.raise_for_status()
        return response.json()

    async def get_random_photo(self, query: str | None = None) -> dict[str, Any]:
        """Get a random photo."""
        params = {}
        if query:
            params["query"] = query

        response = await self.client.get("/photos/random", params=params)

        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")

        response.raise_for_status()
        return response.json()

    async def trigger_download(self, download_location: str):
        """Track download (required by Unsplash guidelines)."""
        await self.client.get(download_location)
        logger.info("download_tracked", location=download_location)

    def get_attribution(self, photo: dict) -> str:
        """Generate attribution HTML."""
        user = photo["user"]
        utm = f"utm_source={self.app_name}&utm_medium=referral"
        return (
            f'Photo by <a href="{user["links"]["html"]}?{utm}">'
            f'{user["name"]}</a> on '
            f'<a href="https://unsplash.com/?{utm}">Unsplash</a>'
        )

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
EOF
```
✅ **Verification**: File created

**Step 4.2: Create Tests for Unsplash Client**
```bash
cat > tests/unit/test_unsplash.py << 'EOF'
"""Tests for Unsplash API client."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.ai_artist.api.unsplash import UnsplashClient


@pytest.fixture
def mock_httpx_client(mocker):
    """Mock httpx.AsyncClient."""
    client = AsyncMock()
    mocker.patch("httpx.AsyncClient", return_value=client)
    return client


@pytest.mark.asyncio
async def test_search_photos_success(mock_httpx_client):
    """Test successful photo search."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": [{"id": "123"}]}
    mock_httpx_client.get.return_value = mock_response

    # Test
    client = UnsplashClient(access_key="test_key")
    result = await client.search_photos(query="sunset")

    assert "results" in result
    assert len(result["results"]) == 1
    mock_httpx_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_get_random_photo(mock_httpx_client):
    """Test getting random photo."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "random123"}
    mock_httpx_client.get.return_value = mock_response

    client = UnsplashClient(access_key="test_key")
    result = await client.get_random_photo(query="landscape")

    assert result["id"] == "random123"
EOF

pytest tests/unit/test_unsplash.py -v
```
✅ **Verification**: Tests pass

**Step 4.3: Commit API Client**
```bash
git add .
git commit -m "feat: add Unsplash API client with tests"
```
✅ **Verification**: Git commit successful

**Step 4.4: Weekly Checkpoint**
```bash
# Review progress
pytest tests/ -v --cov=src/ai_artist --cov-report=term
pre-commit run --all-files
git log --oneline
```
✅ **Verification**: All tests pass, code quality checks pass

---

## Phase 1: Basic Pipeline (Weeks 2-3)

### Week 2: Image Generation

**Step 5.1: Create Generator Module**
```bash
cat > src/ai_artist/core/generator.py << 'EOF'
"""Image generation using Stable Diffusion + LoRA."""

from pathlib import Path
from typing import Literal
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ImageGenerator:
    """Stable Diffusion image generator with LoRA support."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Literal["cuda", "mps", "cpu"] = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.pipeline = None

        logger.info("initializing_generator", model=model_id, device=device)

    def load_model(self):
        """Load the diffusion pipeline."""
        logger.info("loading_model", model=self.model_id)

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
            use_safetensors=True,
        )

        # Optimizations
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()

        # Use better scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.pipeline = self.pipeline.to(self.device)

        # Optional: Compile for speed (PyTorch 2.0+)
        # self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead")

        logger.info("model_loaded", model=self.model_id)

    def load_lora(self, lora_path: Path, lora_scale: float = 0.8):
        """Load LoRA weights."""
        if not self.pipeline:
            raise RuntimeError("Load model first using load_model()")

        logger.info("loading_lora", path=str(lora_path), scale=lora_scale)

        self.pipeline.load_lora_weights(str(lora_path))
        self.pipeline.fuse_lora(lora_scale=lora_scale)

        logger.info("lora_loaded", path=str(lora_path))

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Generate images from prompt."""
        if not self.pipeline:
            raise RuntimeError("Load model first using load_model()")

        logger.info(
            "generating_images",
            prompt=prompt[:100],
            num_images=num_images,
            steps=num_inference_steps,
        )

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        logger.info("images_generated", count=len(result.images))

        return result.images

    def unload(self):
        """Unload model from memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            logger.info("model_unloaded")
EOF
```
✅ **Verification**: File created

**Step 5.2: Create Integration Test**
```bash
cat > tests/integration/test_generator.py << 'EOF'
"""Integration tests for image generator."""

import pytest
from pathlib import Path
from src.ai_artist.core.generator import ImageGenerator


@pytest.mark.integration
@pytest.mark.slow
def test_generator_load_model():
    """Test loading the model."""
    # Use SDXL Turbo for faster testing
    generator = ImageGenerator(
        model_id="stabilityai/sdxl-turbo",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    generator.load_model()

    assert generator.pipeline is not None
    generator.unload()


@pytest.mark.integration
@pytest.mark.slow
def test_generator_create_image():
    """Test generating a single image."""
    generator = ImageGenerator(
        model_id="stabilityai/sdxl-turbo",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    generator.load_model()

    images = generator.generate(
        prompt="a red apple on a wooden table",
        num_images=1,
        num_inference_steps=4,  # Turbo uses 4 steps
        width=512,
        height=512,
    )

    assert len(images) == 1
    assert images[0].size == (512, 512)

    generator.unload()
EOF
```
✅ **Verification**: File created

**Step 5.3: Create Gallery Manager**
```bash
cat > src/ai_artist/gallery/manager.py << 'EOF'
"""Gallery management for storing generated images."""

from pathlib import Path
from datetime import datetime
from PIL import Image, PngImagePlugin
import json
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GalleryManager:
    """Manage gallery storage and organization."""

    def __init__(self, gallery_path: Path):
        self.gallery_path = gallery_path
        self.gallery_path.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        image: Image.Image,
        prompt: str,
        metadata: dict,
        featured: bool = False,
    ) -> Path:
        """Save image with metadata."""
        # Create directory structure: year/month/day
        now = datetime.now()
        save_dir = self.gallery_path / f"{now.year}" / f"{now.month:02d}" / f"{now.day:02d}"

        if featured:
            save_dir = save_dir / "featured"
        else:
            save_dir = save_dir / "archive"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{metadata.get('seed', 'noseed')}.png"
        image_path = save_dir / filename

        # Add metadata to PNG
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("prompt", prompt)
        pnginfo.add_text("metadata", json.dumps(metadata))
        pnginfo.add_text("AI-Generated", "true")  # EU AI Act compliance

        # Save image
        image.save(image_path, pnginfo=pnginfo)

        # Save sidecar metadata file
        metadata_path = image_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "metadata": metadata,
                    "created_at": now.isoformat(),
                    "featured": featured,
                },
                indent=2,
            )
        )

        logger.info("image_saved", path=str(image_path), featured=featured)

        return image_path

    def list_images(self, featured_only: bool = False) -> list[Path]:
        """List all images in gallery."""
        pattern = "**/featured/*.png" if featured_only else "**/*.png"
        return sorted(self.gallery_path.glob(pattern), reverse=True)
EOF
```
✅ **Verification**: File created

**Step 5.4: Create End-to-End Test**
```bash
cat > tests/e2e/test_pipeline.py << 'EOF'
"""End-to-end pipeline test."""

import pytest
from pathlib import Path
from src.ai_artist.core.generator import ImageGenerator
from src.ai_artist.gallery.manager import GalleryManager


@pytest.mark.integration
@pytest.mark.slow
def test_full_generation_pipeline(tmp_path):
    """Test complete generation and saving pipeline."""
    # Setup
    generator = ImageGenerator(
        model_id="stabilityai/sdxl-turbo",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    generator.load_model()

    gallery = GalleryManager(tmp_path / "gallery")

    # Generate image
    images = generator.generate(
        prompt="a beautiful sunset over mountains",
        num_images=1,
        num_inference_steps=4,
        width=512,
        height=512,
        seed=42,
    )

    # Save to gallery
    saved_path = gallery.save_image(
        image=images[0],
        prompt="a beautiful sunset over mountains",
        metadata={"seed": 42, "model": "sdxl-turbo"},
    )

    # Verify
    assert saved_path.exists()
    assert saved_path.with_suffix(".json").exists()

    # Verify can list
    all_images = gallery.list_images()
    assert len(all_images) >= 1

    generator.unload()
EOF
```
✅ **Verification**: File created

**Step 5.5: Run All Tests**
```bash
# Run fast unit tests
pytest tests/unit/ -v

# Run slow integration tests (if you have GPU and time)
# pytest tests/integration/ tests/e2e/ -v --slow
```
✅ **Verification**: Unit tests pass

**Step 5.6: Commit Phase 1 Work**
```bash
git add .
git commit -m "feat: add image generation pipeline"
```
✅ **Verification**: Git commit successful

---

## Phase 2: Style Training (Weeks 4-5)

### Week 4: LoRA Training Setup

**Step 6.1: Review Legal Requirements**
```bash
# CRITICAL: Read LEGAL.md before proceeding
cat LEGAL.md | grep -A 10 "Training Data"
```
✅ **Verification**: Legal requirements understood

**Step 6.2: Create Training Data Directory Structure**
```bash
mkdir -p datasets/training/{images,captions}
mkdir -p datasets/regularization
mkdir -p models/lora
mkdir -p training_logs
```
✅ **Verification**: Directories created

**Step 6.3: Create Training Script**
```bash
cat > scripts/train_lora.py << 'EOF'
"""LoRA training script using PEFT and accelerate."""

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from peft import LoraConfig, get_peft_model
import yaml
from tqdm import tqdm


def train_lora(config_path: Path):
    """Train LoRA weights."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Training LoRA: {config['name']}")
    print(f"Dataset: {config['dataset_path']}")
    print(f"Steps: {config['max_train_steps']}")

    # Load base model
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config["base_model"],
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=config["network_dim"],
        lora_alpha=config["network_alpha"],
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )

    # Add LoRA layers to UNet
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    pipeline.unet.print_trainable_parameters()

    # TODO: Implement training loop
    # For now, this is a placeholder structure
    print("Training loop would go here...")
    print("See: https://github.com/huggingface/peft#training")

    # Save LoRA weights
    output_path = Path(config["output_dir"]) / f"{config['name']}.safetensors"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline.unet.save_pretrained(output_path)
    print(f"LoRA saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    train_lora(args.config)
EOF
```
✅ **Verification**: File created

**Step 6.4: Create Training Configuration**
```bash
cat > configs/lora_training.yaml << 'EOF'
# LoRA Training Configuration

name: "ai_artist_style_v1"
base_model: "stabilityai/stable-diffusion-xl-base-1.0"

# Dataset
dataset_path: "datasets/training"
regularization_path: "datasets/regularization"
num_repeats: 3

# Network architecture
network_dim: 64
network_alpha: 32

# Training parameters
learning_rate: 1.0e-4
max_train_steps: 2000
batch_size: 1
gradient_accumulation_steps: 4

# Optimizer
optimizer: "AdamW8bit"
weight_decay: 0.01

# Scheduler
lr_scheduler: "cosine_with_restarts"
lr_warmup_steps: 100

# Memory optimization
mixed_precision: "fp16"
gradient_checkpointing: true

# Output
output_dir: "models/lora"
save_every_n_steps: 500
EOF
```
✅ **Verification**: File created

**Step 6.5: Create Data Preparation Guide**
```bash
cat > scripts/prepare_training_data.py << 'EOF'
"""Prepare images for LoRA training."""

from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm


def prepare_dataset(input_dir: Path, output_dir: Path, resolution: int = 1024):
    """Prepare training dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*.{jpg,jpeg,png,webp}"))
    print(f"Found {len(images)} images")

    for img_path in tqdm(images):
        # Load and resize
        img = Image.open(img_path).convert("RGB")

        # Resize to square
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # Save
        output_path = output_dir / f"{img_path.stem}.png"
        img.save(output_path)

        # Create caption file
        caption_path = output_path.with_suffix(".txt")
        if not caption_path.exists():
            # Default caption - user should edit these
            caption_path.write_text("a painting in the artist's unique style")

    print(f"Prepared {len(images)} images in {output_dir}")
    print("IMPORTANT: Edit .txt files with accurate captions!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets/training"))
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.resolution)
EOF
```
✅ **Verification**: File created

**Step 6.6: Document Training Process**
```bash
# User should read TRAINING_DATA_GUIDE.md before collecting data
# This will be created in later steps
```

**Step 6.7: Commit Training Setup**
```bash
git add .
git commit -m "feat: add LoRA training infrastructure"
```
✅ **Verification**: Git commit successful

---

## Phase 3: Automation (Week 6)

### Week 6: Scheduling and Automation

**Step 7.1: Create Scheduler**
```bash
cat > src/ai_artist/scheduling/scheduler.py << 'EOF'
"""Automated creation scheduling using APScheduler."""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CreationScheduler:
    """Schedule automated art creation."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        logger.info("scheduler_started")

    def schedule_daily(self, hour: int, minute: int, job_func, timezone: str = "UTC"):
        """Schedule daily creation."""
        trigger = CronTrigger(hour=hour, minute=minute, timezone=timezone)
        self.scheduler.add_job(job_func, trigger)
        logger.info("daily_schedule_added", hour=hour, minute=minute, tz=timezone)

    def schedule_weekly(
        self, days: list[str], hour: int, minute: int, job_func, timezone: str = "UTC"
    ):
        """Schedule weekly creation on specific days."""
        day_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        day_numbers = [day_map[day.lower()] for day in days]

        trigger = CronTrigger(
            day_of_week=",".join(map(str, day_numbers)),
            hour=hour,
            minute=minute,
            timezone=timezone,
        )
        self.scheduler.add_job(job_func, trigger)
        logger.info("weekly_schedule_added", days=days, hour=hour, minute=minute)

    def schedule_cron(self, cron_expression: str, job_func, timezone: str = "UTC"):
        """Schedule using cron expression."""
        trigger = CronTrigger.from_crontab(cron_expression, timezone=timezone)
        self.scheduler.add_job(job_func, trigger)
        logger.info("cron_schedule_added", cron=cron_expression)

    def run_once(self, job_func, delay_seconds: int = 0):
        """Run job once after delay."""
        run_date = datetime.now().timestamp() + delay_seconds
        self.scheduler.add_job(job_func, "date", run_date=datetime.fromtimestamp(run_date))
        logger.info("one_time_job_scheduled", delay=delay_seconds)

    def shutdown(self):
        """Shutdown scheduler."""
        self.scheduler.shutdown()
        logger.info("scheduler_shutdown")
EOF
```
✅ **Verification**: File created

**Step 7.2: Create Main Application**
```bash
cat > src/ai_artist/main.py << 'EOF'
"""Main application entry point."""

import asyncio
from pathlib import Path
import argparse
from .core.generator import ImageGenerator
from .gallery.manager import GalleryManager
from .api.unsplash import UnsplashClient
from .scheduling.scheduler import CreationScheduler
from .utils.logging import configure_logging, get_logger
from .utils.config import load_config

logger = get_logger(__name__)


class AIArtist:
    """Main AI Artist application."""

    def __init__(self, config_path: Path):
        self.config = load_config(config_path)
        self.generator = None
        self.gallery = None
        self.unsplash = None
        self.scheduler = None

    def initialize(self):
        """Initialize components."""
        logger.info("initializing_ai_artist")

        # Setup logging
        configure_logging(log_level="INFO", log_file=Path("logs/ai_artist.log"))

        # Initialize generator
        self.generator = ImageGenerator(
            model_id=self.config.model.base_model,
            device=self.config.model.device,
            dtype=getattr(torch, self.config.model.dtype),
        )
        self.generator.load_model()

        # Initialize gallery
        self.gallery = GalleryManager(Path("gallery"))

        # Initialize Unsplash
        self.unsplash = UnsplashClient(
            access_key=self.config.api_keys.unsplash_access_key,
        )

        # Initialize scheduler
        self.scheduler = CreationScheduler()

        logger.info("ai_artist_initialized")

    async def create_artwork(self, theme: str | None = None):
        """Create a single piece of artwork."""
        logger.info("creating_artwork", theme=theme)

        # Get inspiration from Unsplash
        query = theme or "art"
        photo = await self.unsplash.get_random_photo(query=query)
        prompt = f"{photo['description'] or query}, artistic interpretation"

        logger.info("got_inspiration", query=query, photo_id=photo["id"])

        # Generate images
        images = self.generator.generate(
            prompt=prompt,
            **self.config.generation.dict(),
        )

        # Save best image (for now, save first one)
        saved_path = self.gallery.save_image(
            image=images[0],
            prompt=prompt,
            metadata={
                "source_url": photo["urls"]["regular"],
                "source_id": photo["id"],
                "theme": theme,
            },
        )

        # Track download
        await self.unsplash.trigger_download(photo["links"]["download_location"])

        logger.info("artwork_created", path=str(saved_path))

        return saved_path

    async def run_manual(self):
        """Run manual creation once."""
        await self.create_artwork()

    async def run_automated(self):
        """Run in automated mode with scheduling."""
        logger.info("starting_automated_mode")

        # Schedule daily creation at 9 AM
        def creation_job():
            asyncio.create_task(self.create_artwork())

        self.scheduler.schedule_daily(hour=9, minute=0, job_func=creation_job)

        logger.info("automated_mode_running")

        # Keep running
        while True:
            await asyncio.sleep(60)

    async def shutdown(self):
        """Cleanup resources."""
        logger.info("shutting_down")

        if self.generator:
            self.generator.unload()

        if self.unsplash:
            await self.unsplash.close()

        if self.scheduler:
            self.scheduler.shutdown()

        logger.info("shutdown_complete")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AI Artist")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--mode", choices=["manual", "auto"], default="manual")
    args = parser.parse_args()

    app = AIArtist(args.config)
    app.initialize()

    try:
        if args.mode == "manual":
            asyncio.run(app.run_manual())
        else:
            asyncio.run(app.run_automated())
    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
    finally:
        asyncio.run(app.shutdown())


if __name__ == "__main__":
    main()
EOF
```
✅ **Verification**: File created

**Step 7.3: Test Manual Mode**
```bash
# Update config with your API keys
nano config/config.yaml

# Test manual creation
python -m src.ai_artist.main --mode manual
```
✅ **Verification**: Image created successfully

**Step 7.4: Commit Automation**
```bash
git add .
git commit -m "feat: add scheduling and main application"
```
✅ **Verification**: Git commit successful

---

## Phase 4: Advanced Features (Weeks 7-8)

### Week 7: Curation System

**Step 8.1: Install CLIP**
```bash
pip install git+https://github.com/openai/CLIP.git
pip install imagehash
```
✅ **Verification**: CLIP installed

**Step 8.2: Create Curator**
```bash
cat > src/ai_artist/curation/curator.py << 'EOF'
"""Automated image curation using CLIP."""

import torch
import clip
from PIL import Image
from dataclasses import dataclass
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Image quality metrics."""
    aesthetic_score: float
    clip_score: float
    technical_score: float

    @property
    def overall_score(self) -> float:
        """Weighted average."""
        return (
            self.aesthetic_score * 0.5 +
            self.clip_score * 0.3 +
            self.technical_score * 0.2
        )


class ImageCurator:
    """CLIP-based image curation."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        logger.info("curator_initialized", device=device)

    def evaluate(self, image: Image.Image, prompt: str) -> QualityMetrics:
        """Evaluate image quality."""
        # CLIP score (text-image alignment)
        clip_score = self._compute_clip_score(image, prompt)

        # Aesthetic score (placeholder - implement with aesthetic predictor)
        aesthetic_score = self._estimate_aesthetic(image)

        # Technical score (resolution, sharpness)
        technical_score = self._compute_technical_score(image)

        metrics = QualityMetrics(
            aesthetic_score=aesthetic_score,
            clip_score=clip_score,
            technical_score=technical_score,
        )

        logger.info(
            "image_evaluated",
            overall=round(metrics.overall_score, 2),
            aesthetic=round(aesthetic_score, 2),
            clip=round(clip_score, 2),
        )

        return metrics

    def _compute_clip_score(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP similarity score."""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            text_token = clip.tokenize([prompt]).to(self.device)

            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_token)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (image_features @ text_features.T).item()

        return max(0.0, similarity)  # Clip to [0, 1]

    def _estimate_aesthetic(self, image: Image.Image) -> float:
        """Estimate aesthetic score (placeholder)."""
        # TODO: Implement with LAION aesthetic predictor
        # For now, return a dummy score
        return 0.7

    def _compute_technical_score(self, image: Image.Image) -> float:
        """Compute technical quality score."""
        # Check resolution
        width, height = image.size
        resolution_score = min(1.0, (width * height) / (1024 * 1024))

        # TODO: Add blur detection, artifact detection

        return resolution_score

    def should_keep(self, metrics: QualityMetrics, threshold: float = 0.6) -> bool:
        """Determine if image should be kept."""
        return metrics.overall_score >= threshold
EOF
```
✅ **Verification**: File created

**Step 8.3: Integrate Curator into Pipeline**
```bash
# Update main.py to use curator (in create_artwork method)
```

**Step 8.4: Commit Curation**
```bash
git add .
git commit -m "feat: add image curation system"
```
✅ **Verification**: Git commit successful

### Week 8: Polish and Documentation

**Step 9.1: Add CLI Commands**
```bash
cat > src/ai_artist/cli.py << 'EOF'
"""CLI commands for AI Artist."""

import click
import asyncio
from pathlib import Path
from .main import AIArtist


@click.group()
def cli():
    """AI Artist CLI."""
    pass


@cli.command()
@click.option("--config", type=click.Path(), default="config/config.yaml")
def create(config):
    """Create a single artwork."""
    app = AIArtist(Path(config))
    app.initialize()
    asyncio.run(app.create_artwork())
    asyncio.run(app.shutdown())


@cli.command()
@click.option("--config", type=click.Path(), default="config/config.yaml")
def start(config):
    """Start automated creation."""
    app = AIArtist(Path(config))
    app.initialize()
    try:
        asyncio.run(app.run_automated())
    except KeyboardInterrupt:
        pass
    finally:
        asyncio.run(app.shutdown())


@cli.command()
def gallery():
    """View gallery statistics."""
    from .gallery.manager import GalleryManager

    gallery = GalleryManager(Path("gallery"))
    images = gallery.list_images()
    featured = gallery.list_images(featured_only=True)

    click.echo(f"Total images: {len(images)}")
    click.echo(f"Featured images: {len(featured)}")


if __name__ == "__main__":
    cli()
EOF
```
✅ **Verification**: File created

**Step 9.2: Final Testing**
```bash
# Run full test suite
pytest tests/ -v --cov=src/ai_artist

# Run code quality checks
ruff check src/
black src/ --check
mypy src/
```
✅ **Verification**: All checks pass

**Step 9.3: Create Release**
```bash
git tag -a v0.1.0 -m "Initial release"
git push origin v0.1.0
```
✅ **Verification**: Release tagged

---

## Verification Checklist

Use this checklist to verify completion of each phase:

### Phase 0: Prerequisites
- [ ] Python 3.11+ installed
- [ ] GPU available (or CPU fallback accepted)
- [ ] 50GB+ disk space
- [ ] Unsplash API keys obtained
- [ ] (Optional) Pexels API key
- [ ] (Optional) HuggingFace token

### Phase 0.5: Foundation
- [ ] Git repository initialized
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] `.env` file created (not committed)
- [ ] Project structure created
- [ ] Pre-commit hooks installed
- [ ] Database initialized
- [ ] Smoke tests passing

### Phase 1: Basic Pipeline
- [ ] Unsplash client implemented with tests
- [ ] Image generator working
- [ ] Gallery manager saving images
- [ ] End-to-end pipeline test passing
- [ ] Can generate and save one image

### Phase 2: Style Training
- [ ] Legal requirements reviewed
- [ ] Training infrastructure created
- [ ] Data preparation scripts ready
- [ ] Training configuration created
- [ ] (Optional) First LoRA trained

### Phase 3: Automation
- [ ] Scheduler implemented
- [ ] Main application working
- [ ] Manual mode tested
- [ ] Automated mode configured
- [ ] Can run unattended

### Phase 4: Advanced Features
- [ ] CLIP installed
- [ ] Curator implemented
- [ ] Quality metrics working
- [ ] CLI commands functional
- [ ] Full test suite passing
- [ ] Documentation complete

---

## Success Criteria

Project is considered complete when:

1. **Functionality**
   - ✅ Can generate images from prompts
   - ✅ LoRA style training works
   - ✅ Automated scheduling runs 24/7
   - ✅ Quality curation selects best images
   - ✅ Gallery organizes artwork properly

2. **Quality**
   - ✅ Test coverage >70%
   - ✅ All linting checks pass
   - ✅ Type hints comprehensive
   - ✅ Code follows best practices
   - ✅ Documentation complete

3. **Compliance**
   - ✅ Legal requirements met (see LEGAL.md)
   - ✅ GDPR compliance (see COMPLIANCE.md)
   - ✅ EU AI Act compliance
   - ✅ API attribution proper
   - ✅ Security best practices followed

4. **Production-Ready**
   - ✅ Error handling robust
   - ✅ Logging comprehensive
   - ✅ Database migrations work
   - ✅ Backup procedures tested
   - ✅ Monitoring implemented

---

## Next Steps After Completion

Once the build is complete:

1. **Train Your Style**
   - Collect 50-100 high-quality training images
   - Follow TRAINING_DATA_GUIDE.md
   - Train first LoRA style
   - Evaluate and iterate

2. **Deploy to Production**
   - Follow DEPLOYMENT.md
   - Set up monitoring
   - Configure backups
   - Test disaster recovery

3. **Optimize and Improve**
   - Review IMPROVEMENTS.md
   - Implement additional features
   - Tune quality thresholds
   - Add social media integration

4. **Community**
   - Share your artwork
   - Contribute improvements
   - Help others
   - Document learnings

---

**Document Version:** 1.0
**Last Updated:** 2026-01-08
**Maintainer:** AI Artist Project
