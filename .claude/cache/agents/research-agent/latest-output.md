# Research Report: AI Art Generation Best Practices 2026
Generated: 2026-01-08

## Executive Summary

This report covers best practices for AI art generation in 2026, including Stable Diffusion and LoRA training, image quality metrics, MLOps tooling, API patterns, Python development standards, and database management. The field has matured significantly with FLUX emerging as a strong competitor to Stable Diffusion, memory optimization making LoRA training accessible on consumer hardware, and modern Python tooling (uv) becoming the standard for project management.

## Research Questions

1. Stable Diffusion & LoRA Training (2026)
2. AI Image Quality & Curation
3. MLOps for AI Art Projects
4. API Best Practices
5. Python Best Practices (2026)
6. Database & Storage

---

## 1. Stable Diffusion & LoRA Training (2026)

### Current Model Landscape

**Model Comparison 2026:**

| Model | Best For | VRAM Required | Ecosystem |
|-------|----------|---------------|-----------|
| **FLUX.1** | Photorealism, anatomical accuracy, complex prompts | 16GB+ | Growing |
| **Stable Diffusion 3.5** | Artistic styles, anime, vibrant colors | 12GB+ | Mature |
| **SDXL** | General purpose, wide community support | 8-16GB | Very mature |
| **SD 1.5** | Legacy support, low VRAM situations | 4-8GB | Legacy |

**Recommendations:**
- For photorealism and accurate anatomy: **FLUX**
- For artistic/anime styles: **Stable Diffusion 3.5**
- For balanced quality and ecosystem support: **SDXL**
- For production with limited resources: **SDXL with optimizations**

### LoRA Training Best Practices (2026)

**Recommended Training Parameters for SDXL:**

```python
# kohya_ss / sd-scripts configuration
training_config = {
    # Network Architecture
    "network_module": "networks.lora",
    "network_dim": 64,          # Rank (32-128 typical)
    "network_alpha": 32,        # Often rank/2 or equal to rank

    # Learning Rates
    "learning_rate": 1e-4,
    "text_encoder_lr": 0,       # Disable for SDXL
    "unet_lr": 1e-4,

    # Optimizer
    "optimizer_type": "AdamW8bit",  # Memory efficient
    "optimizer_args": ["weight_decay=0.01"],

    # Training Schedule
    "max_train_steps": 2000,
    "lr_scheduler": "cosine_with_restarts",
    "lr_warmup_steps": 100,

    # Memory Optimization (Critical for SDXL)
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,
    "fused_backward_pass": True,  # SDXL specific

    # Quality
    "min_snr_gamma": 5,  # Improves convergence
    "noise_offset": 0.1,
}
```

**Dataset Preparation:**

```python
# Dataset requirements for quality LoRA training
dataset_requirements = {
    "image_count": "50-100 images (minimum 15)",
    "resolution": "1024x1024 for SDXL, 512x512 for SD1.5",
    "quality": "High-resolution, clear, focused on subject",
    "variety": "Multiple angles, lighting conditions, backgrounds",
    "captions": "Detailed, consistent captioning for each image",
}

# Folder structure
# dataset/
#   3_concept/        # 3 repeats per image
#     image1.png
#     image1.txt      # Caption file
#     image2.png
#     image2.txt
```

### Memory Optimization Techniques

```python
# Memory-efficient training configuration
memory_optimization = {
    # Essential for <16GB VRAM
    "gradient_checkpointing": True,  # -30% VRAM, +22% time
    "mixed_precision": "fp16",

    # SDXL-specific optimizations
    "fused_backward_pass": True,     # --fused_backward_pass
    "fused_optimizer_groups": 8,     # Alternative to fused_backward

    # Batch size adjustments
    "train_batch_size": 1,           # Reduce for low VRAM
    "gradient_accumulation_steps": 4, # Simulate larger batches

    # Advanced: LoRA+ for better efficiency
    "loraplus_lr_ratio": 16,
}

# Hardware requirements by model
hardware_requirements = {
    "SD 1.5 LoRA": "8GB VRAM minimum",
    "SDXL LoRA": "12GB comfortable, 8GB possible with optimization",
    "FLUX LoRA": "16GB+ recommended",
}
```

### Speed Optimizations for Inference

```python
import torch
from diffusers import DiffusionPipeline

# Inference optimization techniques
def create_optimized_pipeline(model_id: str, device: str = "cuda"):
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Half precision
        variant="fp16",
        use_safetensors=True,
    )

    # Enable memory-efficient attention
    pipe.enable_xformers_memory_efficient_attention()

    # VAE slicing for large images
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # Model CPU offload for limited VRAM
    # pipe.enable_model_cpu_offload()  # Use if needed

    # Sequential CPU offload (more aggressive)
    # pipe.enable_sequential_cpu_offload()

    # Compile for speed (PyTorch 2.0+)
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

    return pipe.to(device)

# Batch generation for efficiency
async def generate_batch(
    pipe,
    prompts: list[str],
    batch_size: int = 4,
    **kwargs
):
    """Generate images in batches for throughput."""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        images = pipe(batch, **kwargs).images
        results.extend(images)
    return results
```

**Sources:**
- [LoRA Training 2025: Ultimate Guide](https://sanj.dev/post/lora-training-2025-ultimate-guide)
- [SDXL LoRA Training Guide](https://medium.com/@guillaume.bieler/a-comprehensive-guide-to-training-a-stable-diffusion-xl-lora-optimal-settings-dataset-building-844113a6d5b3)
- [Kohya SS Complete Guide 2025](https://apatero.com/blog/kohya-ss-lora-training-complete-guide-2025)
- [FLUX vs Stable Diffusion 2026 Comparison](https://pxz.ai/blog/flux-vs-stable-diffusion:-technical-&-real-world-comparison-2026)
- [Best SD Models 2026](https://www.cubix.co/blog/best-model-for-stable-diffusion/)

---

## 2. AI Image Quality & Curation

### CLIP Aesthetic Scoring

```python
import torch
import clip
from PIL import Image
from pathlib import Path

class AestheticScorer:
    """CLIP-based aesthetic scoring for AI-generated images."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)

        # Load aesthetic predictor (LAION model)
        self.aesthetic_model = self._load_aesthetic_predictor()

    def _load_aesthetic_predictor(self):
        """Load the LAION aesthetic predictor MLP."""
        # Download from: github.com/christophschuhmann/improved-aesthetic-predictor
        from aesthetic_predictor import MLP

        model = MLP(768)  # CLIP embedding size
        model.load_state_dict(
            torch.load("sac+logos+ava1-l14-linearMSE.pth", map_location=self.device)
        )
        return model.to(self.device).eval()

    def score(self, image: Image.Image) -> float:
        """Return aesthetic score (1-10 scale)."""
        with torch.no_grad():
            # Get CLIP embedding
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            # Predict aesthetic score
            score = self.aesthetic_model(embedding.float())
            return score.item()

    def batch_score(self, images: list[Image.Image]) -> list[float]:
        """Score multiple images efficiently."""
        with torch.no_grad():
            tensors = torch.stack([
                self.preprocess(img) for img in images
            ]).to(self.device)

            embeddings = self.model.encode_image(tensors)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            scores = self.aesthetic_model(embeddings.float())
            return scores.squeeze().tolist()

# Usage example
scorer = AestheticScorer()
image = Image.open("generated_image.png")
aesthetic_score = scorer.score(image)
print(f"Aesthetic score: {aesthetic_score:.2f}/10")

# Thresholds for curation
QUALITY_THRESHOLDS = {
    "excellent": 7.0,
    "good": 5.5,
    "acceptable": 4.5,
    "reject": 4.5,  # Below this
}
```

### FID Score Benchmarking

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from pathlib import Path

class FIDCalculator:
    """Calculate FID scores for image quality assessment."""

    def __init__(self, feature_dim: int = 2048):
        self.fid = FrechetInceptionDistance(feature=feature_dim, normalize=True)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])

    def calculate(
        self,
        real_images: list[Image.Image],
        generated_images: list[Image.Image]
    ) -> float:
        """Calculate FID between real and generated image sets."""
        # Add real images
        real_tensors = torch.stack([
            self.transform(img) for img in real_images
        ])
        self.fid.update(real_tensors, real=True)

        # Add generated images
        gen_tensors = torch.stack([
            self.transform(img) for img in generated_images
        ])
        self.fid.update(gen_tensors, real=False)

        # Compute FID
        score = self.fid.compute()
        self.fid.reset()
        return score.item()

# FID benchmarking standards (2026)
FID_BENCHMARKS = {
    "excellent": 10.0,    # Comparable to real images
    "good": 30.0,         # High quality generation
    "acceptable": 50.0,   # Usable quality
    "poor": 100.0,        # Needs improvement
}
```

### Automated Curation Pipeline

```python
from dataclasses import dataclass
from PIL import Image
import hashlib

@dataclass
class ImageQualityMetrics:
    aesthetic_score: float
    clip_score: float  # Text-image alignment
    technical_score: float  # Resolution, artifacts
    uniqueness_score: float  # Deduplication hash distance

    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        weights = {
            "aesthetic": 0.4,
            "clip": 0.3,
            "technical": 0.2,
            "uniqueness": 0.1,
        }
        return (
            self.aesthetic_score * weights["aesthetic"] +
            self.clip_score * weights["clip"] +
            self.technical_score * weights["technical"] +
            self.uniqueness_score * weights["uniqueness"]
        )

class AutomatedCurator:
    """Automated image curation system."""

    def __init__(
        self,
        aesthetic_threshold: float = 5.5,
        clip_threshold: float = 0.25,
        dedup_threshold: float = 0.95,
    ):
        self.aesthetic_scorer = AestheticScorer()
        self.thresholds = {
            "aesthetic": aesthetic_threshold,
            "clip": clip_threshold,
            "dedup": dedup_threshold,
        }
        self.seen_hashes: set[str] = set()

    def _compute_phash(self, image: Image.Image) -> str:
        """Compute perceptual hash for deduplication."""
        import imagehash
        return str(imagehash.phash(image))

    def _check_duplicate(self, image: Image.Image) -> tuple[bool, float]:
        """Check if image is duplicate based on perceptual hash."""
        phash = self._compute_phash(image)

        for seen_hash in self.seen_hashes:
            # Calculate hash distance
            distance = sum(
                c1 != c2 for c1, c2 in zip(phash, seen_hash)
            ) / len(phash)

            if distance < (1 - self.thresholds["dedup"]):
                return True, 1 - distance

        self.seen_hashes.add(phash)
        return False, 1.0

    def evaluate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> tuple[bool, ImageQualityMetrics]:
        """Evaluate image quality and determine if it passes curation."""
        # Aesthetic score
        aesthetic = self.aesthetic_scorer.score(image)

        # CLIP score (text-image alignment)
        clip_score = self._compute_clip_score(image, prompt)

        # Technical quality
        technical = self._compute_technical_score(image)

        # Uniqueness check
        is_dup, uniqueness = self._check_duplicate(image)

        metrics = ImageQualityMetrics(
            aesthetic_score=aesthetic,
            clip_score=clip_score,
            technical_score=technical,
            uniqueness_score=uniqueness,
        )

        # Determine pass/fail
        passes = (
            aesthetic >= self.thresholds["aesthetic"] and
            clip_score >= self.thresholds["clip"] and
            not is_dup
        )

        return passes, metrics
```

**Sources:**
- [LAION Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor)
- [Improved Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [CLIP-AGIQA Paper](https://arxiv.org/html/2408.15098v2)
- [Survey on Quality Metrics for Text-to-Image](https://arxiv.org/html/2403.11821v5)
- [Rethinking FID](https://arxiv.org/html/2401.09603v2)

---

## 3. MLOps for AI Art Projects

### Experiment Tracking Tool Comparison (2026)

| Tool | Best For | Key Features | Cost |
|------|----------|--------------|------|
| **Weights & Biases** | Deep learning, visual data | Image logging, sweeps, reports | Free tier + paid |
| **MLflow** | Structured ML, on-prem | Model registry, deployment | Open source |
| **Neptune.ai** | Team collaboration | Rich metadata, comparisons | Free tier + paid |
| **ClearML** | Full MLOps pipeline | Auto-logging, orchestration | Open source + cloud |

**Recommendation for AI Art:** Weights & Biases - excellent image logging and visualization.

### Weights & Biases Integration

```python
import wandb
from PIL import Image
from pathlib import Path

class ExperimentTracker:
    """W&B experiment tracking for AI art generation."""

    def __init__(
        self,
        project: str = "ai-art",
        entity: str | None = None,
    ):
        self.project = project
        self.entity = entity
        self.run = None

    def start_experiment(
        self,
        name: str,
        config: dict,
        tags: list[str] | None = None,
    ) -> wandb.Run:
        """Start a new experiment run."""
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config,
            tags=tags or [],
        )
        return self.run

    def log_generation(
        self,
        prompt: str,
        image: Image.Image,
        metrics: dict,
        step: int | None = None,
    ):
        """Log a generated image with metrics."""
        self.run.log({
            "generated_image": wandb.Image(image, caption=prompt),
            "prompt": prompt,
            **metrics,
        }, step=step)

    def log_training_step(
        self,
        loss: float,
        lr: float,
        step: int,
        sample_images: list[Image.Image] | None = None,
    ):
        """Log LoRA training progress."""
        log_dict = {
            "train/loss": loss,
            "train/learning_rate": lr,
            "train/step": step,
        }

        if sample_images:
            log_dict["samples"] = [
                wandb.Image(img) for img in sample_images
            ]

        self.run.log(log_dict, step=step)

    def log_model_artifact(
        self,
        model_path: Path,
        name: str,
        metadata: dict,
    ):
        """Log trained model as artifact."""
        artifact = wandb.Artifact(
            name=name,
            type="model",
            metadata=metadata,
        )
        artifact.add_file(str(model_path))
        self.run.log_artifact(artifact)

    def finish(self):
        """End the experiment run."""
        if self.run:
            self.run.finish()

# Usage
tracker = ExperimentTracker(project="lora-training")
tracker.start_experiment(
    name="sdxl-landscape-lora-v1",
    config={
        "model": "SDXL",
        "network_dim": 64,
        "learning_rate": 1e-4,
        "dataset_size": 100,
    },
    tags=["sdxl", "landscape", "production"],
)
```

### Model Versioning with DVC

```yaml
# dvc.yaml - Pipeline definition
stages:
  train_lora:
    cmd: python train.py --config configs/lora.yaml
    deps:
      - configs/lora.yaml
      - dataset/
    params:
      - train.learning_rate
      - train.network_dim
    outs:
      - models/lora/
    metrics:
      - metrics/training.json:
          cache: false

  evaluate:
    cmd: python evaluate.py --model models/lora/
    deps:
      - models/lora/
      - eval_dataset/
    metrics:
      - metrics/evaluation.json:
          cache: false
    plots:
      - plots/quality_scores.csv:
          x: step
          y: aesthetic_score
```

```bash
# DVC commands for model versioning
dvc init
dvc add models/lora_v1.safetensors
dvc push  # Push to remote storage

# Track experiments
dvc exp run --set-param train.learning_rate=1e-4
dvc exp show  # Compare experiments
```

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'configs/**'
      - 'tests/**'
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync

      - name: Run tests
        run: uv run pytest tests/ -v --cov=src/

      - name: Upload coverage
        uses: codecov/codecov-action@v4

  validate-model:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install uv && uv sync

      - name: Validate model config
        run: uv run python scripts/validate_config.py

      - name: Run evaluation on sample
        run: uv run python scripts/evaluate.py --sample-size 10

      - name: Post metrics to PR
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const metrics = JSON.parse(fs.readFileSync('metrics/evaluation.json'));
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Model Evaluation Results\n\n- Aesthetic Score: ${metrics.aesthetic_score}\n- FID: ${metrics.fid_score}`
            });

  deploy:
    runs-on: ubuntu-latest
    needs: [test, validate-model]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Push model to HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          pip install huggingface_hub
          python scripts/upload_to_hf.py
```

**Sources:**
- [W&B vs MLflow vs Neptune Comparison 2026](https://www.index.dev/skill-vs-skill/ai-wandb-vs-mlflow-vs-neptune)
- [27 MLOps Tools for 2026](https://lakefs.io/blog/mlops-tools/)
- [CI/CD for Machine Learning - Made With ML](https://madewithml.com/courses/mlops/cicd/)
- [GitHub Actions for ML](https://github.com/kingabzpro/CICD-for-Machine-Learning)

---

## 4. API Best Practices

### Unsplash API Integration

```python
from dataclasses import dataclass
from typing import Any
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

@dataclass
class UnsplashConfig:
    access_key: str
    app_name: str
    base_url: str = "https://api.unsplash.com"
    requests_per_hour: int = 50  # Demo limit, 5000 for production

class RateLimitExceeded(Exception):
    """Raised when API rate limit is exceeded."""
    pass

class UnsplashClient:
    """Unsplash API client with proper rate limiting and attribution."""

    def __init__(self, config: UnsplashConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Client-ID {config.access_key}"},
            timeout=30.0,
        )
        self._requests_remaining = config.requests_per_hour

    def _check_rate_limit(self, response: httpx.Response):
        """Update rate limit tracking from response headers."""
        self._requests_remaining = int(
            response.headers.get("X-Ratelimit-Remaining", 0)
        )
        if response.status_code == 429:
            raise RateLimitExceeded("Rate limit exceeded")

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
    )
    async def search_photos(
        self,
        query: str,
        page: int = 1,
        per_page: int = 10,
        orientation: str | None = None,
    ) -> dict[str, Any]:
        """Search for photos with retry logic."""
        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
        }
        if orientation:
            params["orientation"] = orientation

        response = await self.client.get("/search/photos", params=params)
        self._check_rate_limit(response)
        response.raise_for_status()
        return response.json()

    async def get_photo(self, photo_id: str) -> dict[str, Any]:
        """Get a single photo by ID."""
        response = await self.client.get(f"/photos/{photo_id}")
        self._check_rate_limit(response)
        response.raise_for_status()
        return response.json()

    async def trigger_download(self, download_location: str):
        """
        REQUIRED: Track download per Unsplash guidelines.
        Call this when user performs download-like action.
        """
        await self.client.get(download_location)

    def get_attribution_html(self, photo: dict) -> str:
        """Generate proper attribution HTML per Unsplash guidelines."""
        user = photo["user"]
        utm = f"utm_source={self.config.app_name}&utm_medium=referral"

        return (
            f'Photo by <a href="{user["links"]["html"]}?{utm}">'
            f'{user["name"]}</a> on '
            f'<a href="https://unsplash.com/?{utm}">Unsplash</a>'
        )

    def get_hotlink_url(self, photo: dict, size: str = "regular") -> str:
        """
        Get hotlinked URL (required by Unsplash guidelines).
        Size options: raw, full, regular, small, thumb
        """
        return photo["urls"][size]

    async def close(self):
        await self.client.aclose()
```

### Comprehensive Retry Logic with Tenacity

```python
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    wait_random,
    before_sleep_log,
    after_log,
)
import logging
import httpx

logger = logging.getLogger(__name__)

# Retry configuration patterns for different scenarios

# Pattern 1: Basic exponential backoff
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def basic_api_call(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

# Pattern 2: Rate limit handling with longer backoff
class RateLimitError(Exception):
    pass

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def rate_limited_api_call(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        response.raise_for_status()
        return response.json()

# Pattern 3: Jitter to prevent thundering herd
@retry(
    wait=wait_fixed(3) + wait_random(0, 2),  # 3-5 seconds with jitter
    stop=stop_after_attempt(3),
)
async def distributed_api_call(url: str) -> dict:
    """Use jitter when multiple clients might retry simultaneously."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

# Pattern 4: Custom retry logic with callback
def should_retry(exception: Exception) -> bool:
    """Custom logic to determine if retry is appropriate."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 5xx errors, not on 4xx
        return 500 <= exception.response.status_code < 600
    return isinstance(exception, (httpx.TimeoutException, httpx.ConnectError))

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def smart_api_call(url: str) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        if not should_retry(e):
            raise  # Don't retry, re-raise immediately
        raise  # Retry
```

### Fallback Strategies

```python
from typing import TypeVar, Callable, Awaitable
from functools import wraps
import asyncio

T = TypeVar("T")

class FallbackChain:
    """Chain multiple data sources with automatic fallback."""

    def __init__(self):
        self.sources: list[tuple[str, Callable[..., Awaitable[T]]]] = []

    def add_source(
        self,
        name: str,
        func: Callable[..., Awaitable[T]],
    ) -> "FallbackChain":
        self.sources.append((name, func))
        return self

    async def execute(self, *args, **kwargs) -> tuple[str, T]:
        """Try sources in order, return first success."""
        errors = []

        for name, func in self.sources:
            try:
                result = await func(*args, **kwargs)
                return name, result
            except Exception as e:
                errors.append((name, e))
                continue

        # All sources failed
        error_summary = "; ".join(
            f"{name}: {type(e).__name__}" for name, e in errors
        )
        raise RuntimeError(f"All sources failed: {error_summary}")

# Usage example
async def get_from_unsplash(query: str) -> list[dict]:
    """Primary source: Unsplash API"""
    # ... implementation
    pass

async def get_from_pexels(query: str) -> list[dict]:
    """Fallback: Pexels API"""
    # ... implementation
    pass

async def get_from_cache(query: str) -> list[dict]:
    """Final fallback: Local cache"""
    # ... implementation
    pass

# Create fallback chain
image_source = (
    FallbackChain()
    .add_source("unsplash", get_from_unsplash)
    .add_source("pexels", get_from_pexels)
    .add_source("cache", get_from_cache)
)

# Use with automatic fallback
source_name, images = await image_source.execute("landscape mountains")
print(f"Got images from: {source_name}")
```

**Sources:**
- [Unsplash API Documentation](https://unsplash.com/documentation)
- [Unsplash API Guidelines](https://help.unsplash.com/en/articles/2511245-unsplash-api-guidelines)
- [Tenacity Documentation](https://tenacity.readthedocs.io/)
- [Python Retry Logic with Tenacity](https://python.useinstructor.com/concepts/retrying/)
- [How to Retry Failed Python Requests 2026](https://www.zenrows.com/blog/python-requests-retry)

---

## 5. Python Best Practices (2026)

### Recommended Project Structure

```
ai-artist/
├── pyproject.toml          # Project configuration (uv)
├── uv.lock                  # Lock file (auto-generated)
├── .python-version          # Python version
├── README.md
│
├── src/
│   └── ai_artist/           # Main package
│       ├── __init__.py
│       ├── py.typed         # PEP 561 marker
│       │
│       ├── core/            # Core business logic
│       │   ├── __init__.py
│       │   ├── generator.py
│       │   └── curator.py
│       │
│       ├── models/          # Data models
│       │   ├── __init__.py
│       │   └── schemas.py
│       │
│       ├── api/             # API layer
│       │   ├── __init__.py
│       │   ├── unsplash.py
│       │   └── routes.py
│       │
│       ├── db/              # Database
│       │   ├── __init__.py
│       │   ├── models.py
│       │   └── migrations/
│       │
│       └── utils/           # Utilities
│           ├── __init__.py
│           └── logging.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Fixtures
│   ├── unit/
│   │   └── test_generator.py
│   ├── integration/
│   │   └── test_api.py
│   └── e2e/
│       └── test_pipeline.py
│
├── configs/
│   ├── base.yaml
│   └── production.yaml
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
└── notebooks/               # Experimentation
    └── exploration.ipynb
```

### Modern pyproject.toml with uv

```toml
[project]
name = "ai-artist"
version = "0.1.0"
description = "AI-powered art generation and curation"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = ["ai", "art", "stable-diffusion", "image-generation"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "torch>=2.2.0",
    "diffusers>=0.27.0",
    "transformers>=4.38.0",
    "accelerate>=0.27.0",
    "safetensors>=0.4.0",
    "pillow>=10.0.0",
    "httpx>=0.27.0",
    "tenacity>=8.2.0",
    "pydantic>=2.5.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
    "pre-commit>=3.6.0",
]
training = [
    "wandb>=0.16.0",
    "bitsandbytes>=0.42.0",
    "xformers>=0.0.24",
]

[project.scripts]
ai-artist = "ai_artist.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

[tool.ruff]
target-version = "py311"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # Pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "ARG", # flake8-unused-arguments
    "SIM", # flake8-simplify
]
ignore = ["E501"]  # Line length handled by formatter

[tool.ruff.isort]
known-first-party = ["ai_artist"]

[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["torch.*", "diffusers.*", "transformers.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
]
```

### Type Hints Best Practices

```python
from typing import TypeVar, Generic, Protocol, Literal
from collections.abc import Sequence, Callable, Awaitable
from dataclasses import dataclass
from pydantic import BaseModel, Field
from PIL import Image
import torch

# Use modern type syntax (Python 3.10+)
type ImageBatch = list[Image.Image]
type Tensor = torch.Tensor

# Protocol for duck typing
class ImageProcessor(Protocol):
    def process(self, image: Image.Image) -> Image.Image: ...
    async def process_async(self, image: Image.Image) -> Image.Image: ...

# Generic types for reusable components
T = TypeVar("T")
ResultT = TypeVar("ResultT")

@dataclass
class Result(Generic[T]):
    """Generic result wrapper with error handling."""
    value: T | None
    error: str | None = None

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(value=value)

    @classmethod
    def err(cls, error: str) -> "Result[T]":
        return cls(value=None, error=error)

# Pydantic models for API schemas
class GenerationRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., min_length=1, max_length=500)
    negative_prompt: str = Field(default="", max_length=500)
    num_images: int = Field(default=1, ge=1, le=4)
    size: Literal["512x512", "768x768", "1024x1024"] = "1024x1024"
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)

    model_config = {"extra": "forbid"}

class GenerationResponse(BaseModel):
    """Response model for image generation."""
    request_id: str
    images: list[str]  # Base64 encoded or URLs
    metrics: dict[str, float]
    generation_time_ms: int

# Callable type hints for callbacks
type ProgressCallback = Callable[[int, int], None]
type AsyncProgressCallback = Callable[[int, int], Awaitable[None]]

async def generate_with_progress(
    prompt: str,
    callback: AsyncProgressCallback | None = None,
) -> list[Image.Image]:
    """Generate images with optional progress callback."""
    steps = 50
    images: list[Image.Image] = []

    for step in range(steps):
        # ... generation logic
        if callback:
            await callback(step, steps)

    return images
```

### Async/Await Patterns for API Calls

```python
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator
import httpx

class AsyncAPIClient:
    """Async API client with connection pooling and proper lifecycle."""

    def __init__(self, base_url: str, max_connections: int = 100):
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None
        self._max_connections = max_connections

    async def __aenter__(self) -> "AsyncAPIClient":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            limits=httpx.Limits(max_connections=self._max_connections),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not connected. Use 'async with' or call connect()")
        return self._client

    async def get(self, path: str, **kwargs) -> httpx.Response:
        return await self.client.get(path, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        return await self.client.post(path, **kwargs)

# Concurrent request pattern
async def fetch_multiple(
    client: AsyncAPIClient,
    urls: list[str],
    max_concurrent: int = 10,
) -> list[dict]:
    """Fetch multiple URLs with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str) -> dict:
        async with semaphore:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks, return_exceptions=True)

# Async generator for streaming results
async def generate_images_stream(
    prompts: list[str],
    batch_size: int = 4,
) -> AsyncIterator[Image.Image]:
    """Stream generated images as they complete."""
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        # Generate batch
        images = await generate_batch(batch)
        for image in images:
            yield image

# Usage
async def main():
    async with AsyncAPIClient("https://api.example.com") as client:
        results = await fetch_multiple(
            client,
            ["/photo/1", "/photo/2", "/photo/3"],
            max_concurrent=5,
        )

    # Stream processing
    async for image in generate_images_stream(prompts):
        # Process each image as it's generated
        pass
```

### Testing Strategies for ML Code

```python
# tests/conftest.py
import pytest
from PIL import Image
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock

@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample image for testing."""
    return Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )

@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    client = AsyncMock()
    client.get.return_value.json.return_value = {"results": []}
    return client

@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a temporary dataset for testing."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    for i in range(5):
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        img.save(dataset_dir / f"image_{i}.png")
        (dataset_dir / f"image_{i}.txt").write_text(f"Test caption {i}")

    return dataset_dir

# tests/unit/test_curator.py
import pytest
from ai_artist.core.curator import AutomatedCurator, ImageQualityMetrics

class TestAutomatedCurator:
    """Unit tests for the automated curator."""

    @pytest.fixture
    def curator(self):
        return AutomatedCurator(
            aesthetic_threshold=5.0,
            clip_threshold=0.2,
        )

    def test_quality_metrics_overall_score(self):
        """Test that overall score calculation is correct."""
        metrics = ImageQualityMetrics(
            aesthetic_score=7.0,
            clip_score=0.8,
            technical_score=0.9,
            uniqueness_score=1.0,
        )

        expected = 7.0 * 0.4 + 0.8 * 0.3 + 0.9 * 0.2 + 1.0 * 0.1
        assert abs(metrics.overall_score - expected) < 0.001

    @pytest.mark.parametrize("aesthetic,expected_pass", [
        (6.0, True),
        (5.0, True),
        (4.0, False),
    ])
    def test_aesthetic_threshold(
        self,
        curator,
        sample_image,
        aesthetic,
        expected_pass,
        mocker,
    ):
        """Test that aesthetic threshold is enforced."""
        mocker.patch.object(
            curator.aesthetic_scorer,
            "score",
            return_value=aesthetic,
        )
        mocker.patch.object(
            curator,
            "_compute_clip_score",
            return_value=0.5,
        )
        mocker.patch.object(
            curator,
            "_compute_technical_score",
            return_value=0.8,
        )

        passes, _ = curator.evaluate(sample_image, "test prompt")
        assert passes == expected_pass

# tests/integration/test_generation_pipeline.py
import pytest
from ai_artist.core.generator import ImageGenerator

@pytest.mark.integration
@pytest.mark.slow
class TestGenerationPipeline:
    """Integration tests for the generation pipeline."""

    @pytest.fixture
    def generator(self):
        # Use a smaller model for testing
        return ImageGenerator(model="stabilityai/sdxl-turbo")

    async def test_single_image_generation(self, generator):
        """Test generating a single image."""
        images = await generator.generate(
            prompt="a red apple on a wooden table",
            num_images=1,
        )

        assert len(images) == 1
        assert images[0].size == (512, 512)

    async def test_batch_generation(self, generator):
        """Test generating multiple images."""
        prompts = [
            "a mountain landscape",
            "a city skyline at night",
        ]

        images = await generator.generate_batch(prompts)
        assert len(images) == 2
```

**Sources:**
- [Python UV: The Ultimate Guide](https://www.datacamp.com/tutorial/python-uv)
- [Managing Python Projects With uv](https://realpython.com/python-uv/)
- [FastAPI Project Structure Best Practices](https://www.sourcetrail.com/python/fastapi-project-structure-and-best-practice-guides/)
- [PyTest for Machine Learning Tutorial](https://towardsdatascience.com/pytest-for-machine-learning-a-simple-example-based-tutorial-a3df3c58cf8/)
- [pytest-mock Tutorial](https://www.datacamp.com/tutorial/pytest-mock)

---

## 6. Database & Storage

### SQLite Best Practices for ML Projects

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from datetime import datetime
from pathlib import Path

Base = declarative_base()

class GeneratedImage(Base):
    """Model for tracking generated images."""
    __tablename__ = "generated_images"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False, index=True)
    prompt = Column(String, nullable=False)
    negative_prompt = Column(String, default="")
    model_id = Column(String, nullable=False, index=True)

    # Generation parameters (JSON for flexibility)
    generation_params = Column(JSON, default=dict)

    # Quality metrics
    aesthetic_score = Column(Float)
    clip_score = Column(Float)
    fid_score = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Status tracking
    status = Column(String, default="pending", index=True)  # pending, curated, rejected

class ModelVersion(Base):
    """Model for tracking LoRA versions."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    path = Column(String, nullable=False)

    # Training metadata
    training_config = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        {"sqlite_autoincrement": True},
    )

def create_database(db_path: Path) -> sessionmaker:
    """Create SQLite database with proper settings for ML workloads."""
    # Use absolute path and create parent directories
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        # Connection pool settings for SQLite
        connect_args={
            "check_same_thread": False,
            "timeout": 30,
        },
        # WAL mode for better concurrent access
        pool_pre_ping=True,
        # For testing, use StaticPool
        # poolclass=StaticPool,
    )

    # Enable WAL mode for better performance
    with engine.connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")

    Base.metadata.create_all(engine)

    return sessionmaker(bind=engine)

# Repository pattern for clean data access
class ImageRepository:
    """Repository for generated image operations."""

    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def add(self, image: GeneratedImage) -> GeneratedImage:
        with self.session_factory() as session:
            session.add(image)
            session.commit()
            session.refresh(image)
            return image

    def get_by_id(self, image_id: int) -> GeneratedImage | None:
        with self.session_factory() as session:
            return session.query(GeneratedImage).filter_by(id=image_id).first()

    def get_uncurated(self, limit: int = 100) -> list[GeneratedImage]:
        with self.session_factory() as session:
            return (
                session.query(GeneratedImage)
                .filter_by(status="pending")
                .order_by(GeneratedImage.created_at.desc())
                .limit(limit)
                .all()
            )

    def update_quality_scores(
        self,
        image_id: int,
        aesthetic_score: float,
        clip_score: float,
    ) -> None:
        with self.session_factory() as session:
            session.query(GeneratedImage).filter_by(id=image_id).update({
                "aesthetic_score": aesthetic_score,
                "clip_score": clip_score,
            })
            session.commit()

    def mark_curated(self, image_id: int, accepted: bool) -> None:
        status = "curated" if accepted else "rejected"
        with self.session_factory() as session:
            session.query(GeneratedImage).filter_by(id=image_id).update({
                "status": status,
            })
            session.commit()
```

### Alembic Migrations for SQLite

```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os

config = context.config

# Get database URL from environment
database_url = os.getenv("DATABASE_URL", "sqlite:///./data/ai_artist.db")
config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from ai_artist.db.models import Base
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # SQLite-specific: enable batch mode
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Disable foreign keys during migration for SQLite
        if connection.dialect.name == "sqlite":
            connection.execute("PRAGMA foreign_keys=OFF")

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite-specific: enable batch mode for ALTER TABLE
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()

        # Re-enable foreign keys
        if connection.dialect.name == "sqlite":
            connection.execute("PRAGMA foreign_keys=ON")

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

```python
# alembic/versions/001_initial_schema.py
"""Initial schema

Revision ID: 001
Create Date: 2026-01-08
"""
from alembic import op
import sqlalchemy as sa

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'generated_images',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('prompt', sa.String(), nullable=False),
        sa.Column('negative_prompt', sa.String(), default=''),
        sa.Column('model_id', sa.String(), nullable=False),
        sa.Column('generation_params', sa.JSON(), default=dict),
        sa.Column('aesthetic_score', sa.Float()),
        sa.Column('clip_score', sa.Float()),
        sa.Column('fid_score', sa.Float()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('status', sa.String(), default='pending'),
    )

    # Create indexes
    op.create_index('ix_generated_images_filename', 'generated_images', ['filename'])
    op.create_index('ix_generated_images_model_id', 'generated_images', ['model_id'])
    op.create_index('ix_generated_images_created_at', 'generated_images', ['created_at'])
    op.create_index('ix_generated_images_status', 'generated_images', ['status'])

def downgrade() -> None:
    op.drop_table('generated_images')
```

### Backup and Recovery

```python
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseBackup:
    """SQLite backup and recovery utilities."""

    def __init__(self, db_path: Path, backup_dir: Path):
        self.db_path = db_path
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, label: str = "") -> Path:
        """Create a backup using SQLite's backup API."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        if label:
            backup_name += f"_{label}"
        backup_name += ".db"

        backup_path = self.backup_dir / backup_name

        # Use SQLite backup API for consistent backup
        source = sqlite3.connect(self.db_path)
        dest = sqlite3.connect(backup_path)

        try:
            source.backup(dest)
            logger.info(f"Created backup: {backup_path}")
        finally:
            source.close()
            dest.close()

        return backup_path

    def restore_from_backup(self, backup_path: Path) -> None:
        """Restore database from backup."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        # Create a safety backup of current database
        safety_backup = self.create_backup(label="pre_restore")
        logger.info(f"Created safety backup: {safety_backup}")

        # Restore
        source = sqlite3.connect(backup_path)
        dest = sqlite3.connect(self.db_path)

        try:
            source.backup(dest)
            logger.info(f"Restored from: {backup_path}")
        finally:
            source.close()
            dest.close()

    def list_backups(self) -> list[Path]:
        """List all available backups."""
        return sorted(
            self.backup_dir.glob("backup_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def cleanup_old_backups(self, keep_count: int = 10) -> None:
        """Remove old backups, keeping the most recent ones."""
        backups = self.list_backups()

        for backup in backups[keep_count:]:
            backup.unlink()
            logger.info(f"Removed old backup: {backup}")

# Automated backup on application events
class BackupScheduler:
    """Schedule automatic backups."""

    def __init__(self, backup_manager: DatabaseBackup):
        self.backup_manager = backup_manager
        self._last_backup = datetime.min

    def maybe_backup(self, min_interval_hours: int = 24) -> Path | None:
        """Create backup if enough time has passed."""
        now = datetime.now()
        hours_since = (now - self._last_backup).total_seconds() / 3600

        if hours_since >= min_interval_hours:
            backup_path = self.backup_manager.create_backup(label="auto")
            self._last_backup = now
            return backup_path

        return None
```

### Efficient Metadata Storage Patterns

```python
from typing import Any
from datetime import datetime
import json
from pathlib import Path

class MetadataStore:
    """Efficient metadata storage with caching."""

    def __init__(self, db_session_factory, cache_size: int = 1000):
        self.session_factory = db_session_factory
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_size = cache_size

    def store_generation_metadata(
        self,
        image_id: str,
        prompt: str,
        params: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Store metadata for a generated image."""
        metadata = {
            "prompt": prompt,
            "params": params,
            "metrics": metrics or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Update cache
        self._update_cache(image_id, metadata)

        # Persist to database
        with self.session_factory() as session:
            # Use upsert pattern
            stmt = """
                INSERT INTO image_metadata (id, data)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET data = excluded.data
            """
            session.execute(stmt, (image_id, json.dumps(metadata)))
            session.commit()

    def get_metadata(self, image_id: str) -> dict[str, Any] | None:
        """Get metadata with cache-first lookup."""
        # Check cache first
        if image_id in self._cache:
            return self._cache[image_id]

        # Load from database
        with self.session_factory() as session:
            result = session.execute(
                "SELECT data FROM image_metadata WHERE id = ?",
                (image_id,)
            ).fetchone()

            if result:
                metadata = json.loads(result[0])
                self._update_cache(image_id, metadata)
                return metadata

        return None

    def _update_cache(self, key: str, value: dict[str, Any]) -> None:
        """Update cache with LRU eviction."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO, could use OrderedDict for LRU)
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[key] = value

    def bulk_store(self, items: list[tuple[str, dict[str, Any]]]) -> None:
        """Efficiently store multiple metadata entries."""
        with self.session_factory() as session:
            session.execute(
                """
                INSERT INTO image_metadata (id, data)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET data = excluded.data
                """,
                [(id, json.dumps(data)) for id, data in items]
            )
            session.commit()
```

**Sources:**
- [Alembic Database Migrations Guide](https://medium.com/@tejpal.abhyuday/alembic-database-migrations-the-complete-developers-guide-d3fc852a6a9e)
- [Running Batch Migrations for SQLite - Alembic Docs](https://alembic.sqlalchemy.org/en/latest/batch.html)
- [Best Practices for Alembic Schema Migration](https://www.pingcap.com/article/best-practices-alembic-schema-migration/)
- [Building Flexible Databases with SQLAlchemy and Alembic](https://www.kubeblogs.com/build-databases-with-sqlalchemy-and-alembic/)

---

## Recommendations

### Immediate Actions

1. **Model Selection**: Start with SDXL for balanced quality and ecosystem support. Consider FLUX for photorealistic needs.

2. **Project Setup**: Use uv with pyproject.toml for dependency management. Follow the recommended project structure.

3. **Experiment Tracking**: Set up Weights & Biases for visual experiment tracking and image logging.

4. **Database**: Use SQLite with WAL mode and Alembic for migrations. Enable batch mode for SQLite compatibility.

### Development Workflow

1. **Type Safety**: Use strict mypy with Pydantic models for runtime validation.

2. **Testing**: Implement three-tier testing (unit, integration, e2e) with pytest and pytest-mock.

3. **API Resilience**: Use tenacity for retry logic with exponential backoff and jitter.

4. **CI/CD**: Set up GitHub Actions for testing, model validation, and deployment.

### Quality Assurance

1. **Image Curation**: Implement CLIP aesthetic scoring with threshold-based filtering.

2. **FID Monitoring**: Track FID scores against reference datasets to monitor model quality.

3. **Automated Curation**: Build a pipeline combining aesthetic, CLIP, and deduplication checks.

---

## Open Questions

1. **FLUX LoRA Training**: Best practices for FLUX-specific LoRA training are still emerging in early 2026. Monitor kohya-ss/sd-scripts updates.

2. **Multi-GPU Training**: Distributed training setups for large-scale LoRA fine-tuning need project-specific configuration.

3. **Production Deployment**: Specific deployment patterns depend on scale requirements (serverless vs. dedicated GPU).

4. **Video Generation**: Emerging models like Sora and open-source alternatives may change best practices later in 2026.

---

## Sources Summary

### Stable Diffusion & LoRA
- [LoRA Training 2025: Ultimate Guide](https://sanj.dev/post/lora-training-2025-ultimate-guide)
- [SDXL LoRA Training Guide](https://medium.com/@guillaume.bieler/a-comprehensive-guide-to-training-a-stable-diffusion-xl-lora-optimal-settings-dataset-building-844113a6d5b3)
- [Kohya SS Complete Guide 2025](https://apatero.com/blog/kohya-ss-lora-training-complete-guide-2025)
- [FLUX vs Stable Diffusion 2026](https://pxz.ai/blog/flux-vs-stable-diffusion:-technical-&-real-world-comparison-2026)
- [Best AI Image Generators 2026](https://wavespeed.ai/blog/posts/best-ai-image-generators-2026/)

### Image Quality
- [LAION Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor)
- [Improved Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [Survey on Quality Metrics](https://arxiv.org/html/2403.11821v5)

### MLOps
- [W&B vs MLflow vs Neptune 2026](https://www.index.dev/skill-vs-skill/ai-wandb-vs-mlflow-vs-neptune)
- [27 MLOps Tools for 2026](https://lakefs.io/blog/mlops-tools/)
- [CI/CD for ML](https://madewithml.com/courses/mlops/cicd/)

### Python & APIs
- [Python UV Guide](https://www.datacamp.com/tutorial/python-uv)
- [Unsplash API Documentation](https://unsplash.com/documentation)
- [Tenacity Documentation](https://tenacity.readthedocs.io/)
- [PyTest for ML](https://towardsdatascience.com/pytest-for-machine-learning-a-simple-example-based-tutorial-a3df3c58cf8/)

### Database
- [Alembic Migrations Guide](https://medium.com/@tejpal.abhyuday/alembic-database-migrations-the-complete-developers-guide-d3fc852a6a9e)
- [Alembic SQLite Batch Operations](https://alembic.sqlalchemy.org/en/latest/batch.html)
