"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Literal

import torch
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Model configuration."""

    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    dtype: Literal["float16", "float32"] = "float16"
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    lora_path: str | None = None  # Path to trained LoRA weights
    lora_scale: float = 0.8  # LoRA strength (0.0-1.0)


class GenerationConfig(BaseModel):
    """Generation parameters."""

    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)
    num_inference_steps: int = Field(30, ge=10, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    num_variations: int = Field(3, ge=1, le=10)
    negative_prompt: str = "blurry, low quality, distorted, ugly, deformed"


class APIKeysConfig(BaseModel):
    """API keys configuration."""

    unsplash_access_key: str
    unsplash_secret_key: str
    pexels_api_key: str | None = None
    hf_token: str | None = None


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = "sqlite:///./data/ai_artist.db"


class Config(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(extra="allow", env_file=".env", env_nested_delimiter="__")  # type: ignore[misc]

    model: ModelConfig = Field(default_factory=ModelConfig)  # type: ignore[arg-type]
    generation: GenerationConfig = Field(default_factory=GenerationConfig)  # type: ignore[arg-type]
    api_keys: APIKeysConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)  # type: ignore[arg-type]


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return Config(**data)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    return torch.float16 if dtype_str == "float16" else torch.float32
