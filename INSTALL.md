# Installation Guide

## Quick Start

### 1. Run the Setup Script (Recommended)

```bash
cd /Volumes/LizsDisk/ai-artist
bash scripts/setup_project.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Create necessary directories
- Copy example configuration
- Initialize database
- Set up pre-commit hooks

### 2. Configure API Keys

```bash
# Edit the config file
nano config/config.yaml

# Add your API keys:
# - Unsplash Access Key
# - Unsplash Secret Key
# - (Optional) Pexels API Key
# - (Optional) HuggingFace Token
```

Get your API keys:
- **Unsplash**: https://unsplash.com/developers
- **Pexels**: https://www.pexels.com/api/
- **HuggingFace**: https://huggingface.co/settings/tokens

### 3. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Or with coverage
pytest --cov=src/ai_artist
```

### 5. Test Image Generation

```bash
# Quick test with small model
python scripts/test_generation.py

# Or run manually
python -m src.ai_artist.main --mode manual --theme "sunset"
```

## Manual Installation

If you prefer to install manually:

### Prerequisites

```bash
# Check Python version (3.11+ required)
python3 --version

# Check if pip is installed
python3 -m pip --version
```

### Step 1: Create Virtual Environment

```bash
cd /Volumes/LizsDisk/ai-artist
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install PyTorch

**For NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision torchaudio
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install with development tools
pip install -e ".[dev]"
```

### Step 4: Create Directories

```bash
mkdir -p models/{lora,cache}
mkdir -p gallery
mkdir -p data
mkdir -p logs
mkdir -p datasets/{training,regularization}
```

### Step 5: Configure

```bash
# Copy example config
cp config/config.example.yaml config/config.yaml

# Edit with your API keys
nano config/config.yaml
```

### Step 6: Initialize Database

```bash
alembic upgrade head
```

### Step 7: Install Pre-commit Hooks (Optional)

```bash
pre-commit install
```

## Verify Installation

```bash
# Check imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "from src.ai_artist import __version__; print(f'AI Artist: {__version__}')"

# Run tests
pytest tests/test_smoke.py -v
```

## System Requirements

### Minimum
- Python 3.11+
- 16GB RAM
- 6GB VRAM (NVIDIA RTX 2060, M1 Max)
- 50GB free disk space

### Recommended
- Python 3.11+
- 32GB RAM
- 12GB+ VRAM (NVIDIA RTX 3060, RTX 4070)
- 100GB free disk space

### Supported Platforms
- ✅ macOS (Intel & Apple Silicon)
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+)

## Troubleshooting

### "Module not found" errors

```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### "CUDA not available"

```bash
# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Out of memory" errors

Edit `config/config.yaml`:
```yaml
model:
  device: "cuda"
  dtype: "float16"
  enable_attention_slicing: true
  enable_vae_slicing: true

generation:
  width: 512
  height: 512
```

### Database errors

```bash
# Recreate database
rm data/ai_artist.db
alembic upgrade head
```

## What's Next?

After installation:

1. **Run your first generation**: 
   ```bash
   python -m src.ai_artist.main --mode manual --theme "landscape"
   ```

2. **Check the output**: 
   ```bash
   ls -la gallery/
   ```

3. **Read the documentation**:
   - [QUICKSTART.md](QUICKSTART.md) - Getting started guide
   - [README.md](README.md) - Project overview
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System design

4. **Start Phase 1**: Follow [BUILD_GUIDE.md](BUILD_GUIDE.md) for next steps

## Getting Help

- Check [README_IMPLEMENTATION.md](README_IMPLEMENTATION.md) for implementation status
- Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow

---

**Installation Guide** | Version 1.0 | Last Updated: 2026-01-08

