"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Literal

import torch
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, SecretStr
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
    use_refiner: bool = False
    refiner_model: str = "stabilityai/stable-diffusion-xl-refiner-1.0"


class GenerationConfig(BaseModel):
    """Generation parameters."""

    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)
    num_inference_steps: int = Field(30, ge=10, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    num_variations: int = Field(3, ge=1, le=10)
    negative_prompt: str = "blurry, low quality, distorted, ugly, deformed"


class UpscalingConfig(BaseModel):
    """Upscaling configuration."""

    enabled: bool = False
    model_id: str = "stabilityai/stable-diffusion-x4-upscaler"
    noise_level: int = 20  # 0-100 (20 is good for faithful upscaling)


class ControlNetConfig(BaseModel):
    """ControlNet configuration."""

    enabled: bool = False
    model_id: str = "lllyasviel/sd-controlnet-canny"
    conditioning_scale: float = 1.0
    low_threshold: int = 100  # Canny edge detection low threshold
    high_threshold: int = 200  # Canny edge detection high threshold


class InpaintingConfig(BaseModel):
    """Inpainting configuration."""

    enabled: bool = False
    model_id: str = "runwayml/stable-diffusion-inpainting"


class FaceRestorationConfig(BaseModel):
    """Face restoration configuration."""

    enabled: bool = False
    model: Literal["gfpgan", "codeformer"] = "gfpgan"
    model_path: str | None = None  # Custom model path (optional)


class AutonomyConfig(BaseModel):
    """Autonomy / Feedback Loop configuration."""

    enabled: bool = False
    min_score_threshold: float = (
        0.22  # Minimum CLIP score (0.22 is decent for pure CLIP)
    )
    max_retries: int = 2
    feedback_mode: Literal["simple_retry", "refine_prompt"] = "simple_retry"


class TrendsConfig(BaseModel):
    """Trend analysis configuration."""

    enabled: bool = False
    update_interval_hours: int = 24
    sources: list[str] = ["civitai", "artstation"]


class ModelManagerConfig(BaseModel):
    """Model management configuration."""

    enabled: bool = False
    base_path: str = "models"
    auto_download_trending: bool = False
    max_models: int = 50
    civitai_api_key: str | None = None


class APIKeysConfig(BaseModel):
    """API keys configuration.

    Uses SecretStr to prevent accidental logging of sensitive values.
    Access the actual value with .get_secret_value() method.
    """

    unsplash_access_key: SecretStr
    unsplash_secret_key: SecretStr
    pexels_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    civitai_api_key: SecretStr | None = None


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = "sqlite:///./data/ai_artist.db"


class Config(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        extra="allow", env_file=".env", env_nested_delimiter="__"
    )  # type: ignore[misc]

    model: ModelConfig = Field(default_factory=ModelConfig)  # type: ignore[arg-type]
    generation: GenerationConfig = Field(default_factory=GenerationConfig)  # type: ignore[arg-type]
    upscaling: UpscalingConfig = Field(default_factory=UpscalingConfig)  # type: ignore[arg-type]
    controlnet: ControlNetConfig = Field(default_factory=ControlNetConfig)  # type: ignore[arg-type]
    inpainting: InpaintingConfig = Field(default_factory=InpaintingConfig)  # type: ignore[arg-type]
    face_restoration: FaceRestorationConfig = Field(
        default_factory=FaceRestorationConfig
    )  # type: ignore[arg-type]
    autonomy: AutonomyConfig = Field(default_factory=AutonomyConfig)  # type: ignore[arg-type]
    trends: TrendsConfig = Field(default_factory=TrendsConfig)  # type: ignore[arg-type]
    model_manager: ModelManagerConfig = Field(default_factory=ModelManagerConfig)  # type: ignore[arg-type]
    api_keys: APIKeysConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)  # type: ignore[arg-type]


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file.

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required configuration is missing
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)
    config = Config(**data)

    # Validate required fields
    if not config.api_keys.unsplash_access_key.get_secret_value():
        raise ValueError("unsplash_access_key is required in config")
    if not config.api_keys.unsplash_secret_key.get_secret_value():
        raise ValueError("unsplash_secret_key is required in config")

    return config


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    return torch.float16 if dtype_str == "float16" else torch.float32
