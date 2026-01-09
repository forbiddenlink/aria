# Setup Guide

Complete installation and setup instructions for the AI Artist project.

## Prerequisites

### System Requirements

**Minimum**:
- Python 3.10 or higher
- 16GB RAM
- 6GB VRAM GPU (NVIDIA RTX 2060 or M1 Max)
- 50GB free disk space

**Recommended**:
- Python 3.11
- 32GB RAM
- 12GB+ VRAM GPU (NVIDIA RTX 3060, RTX 4070)
- 100GB free disk space (for models and gallery)

**Supported Platforms**:
- ✅ macOS (Intel & Apple Silicon)
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+)

### Required Accounts

1. **Unsplash Developer Account** (Free)
   - Sign up: https://unsplash.com/developers
   - Create new app to get API keys

2. **Pexels API** (Optional, Free)
   - Sign up: https://www.pexels.com/api/
   - Get API key

3. **Hugging Face** (Free)
   - Sign up: https://huggingface.co/join
   - Get access token (for model downloads)

---

## Installation

### Step 1: Clone or Navigate to Project

```bash
cd ~/Desktop/LizsDisk/ai-artist
```

### Step 2: Create Virtual Environment

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**:
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install PyTorch

Choose the appropriate command for your system:

**NVIDIA GPU (Windows/Linux)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Apple Silicon (M1/M2/M3 Mac)**:
```bash
pip install torch torchvision torchaudio
```

**CPU Only** (Not recommended):
```bash
pip install torch torchvision torchaudio
```

**Verify Installation**:
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Diffusers & Transformers (Stable Diffusion)
- Image processing libraries
- API clients
- Scheduling system
- All utilities

**Installation Time**: 5-15 minutes (depending on connection)

### Step 5: Create Project Structure

```bash
# Create necessary directories
mkdir -p {models/lora,models/checkpoints,gallery,data,logs,config,datasets}

# Create .gitkeep files for empty directories
touch models/lora/.gitkeep gallery/.gitkeep data/.gitkeep logs/.gitkeep
```

### Step 6: Configure Settings

1. **Copy example config**:
```bash
cp config/config.example.yaml config/config.yaml
```

2. **Edit configuration**:
```bash
# Use your preferred editor
nano config/config.yaml
# or
code config/config.yaml
# or
vim config/config.yaml
```

3. **Add API keys**:
```yaml
api_keys:
  unsplash:
    access_key: "your_access_key_here"
    secret_key: "your_secret_key_here"
```

4. **Configure device**:
```yaml
model:
  device: "cuda"  # or "mps" for Mac, "cpu" for CPU
  dtype: "float16"
```

5. **Save and close**

### Step 7: Download Base Model

The first run will automatically download the Stable Diffusion model (~4GB).

**Optional: Pre-download**:
```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    cache_dir="models/cache"
)
print("Model downloaded successfully!")
```

### Step 8: Test Installation

Create a test script:

```bash
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify installation"""

import sys
print("Testing AI Artist setup...\n")

# Test imports
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import diffusers
    print(f"✓ Diffusers {diffusers.__version__}")
except ImportError as e:
    print(f"✗ Diffusers: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("✓ Pillow")
except ImportError as e:
    print(f"✗ Pillow: {e}")
    sys.exit(1)

try:
    import yaml
    print("✓ PyYAML")
except ImportError as e:
    print(f"✗ PyYAML: {e}")
    sys.exit(1)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    print("✓ APScheduler")
except ImportError as e:
    print(f"✗ APScheduler: {e}")
    sys.exit(1)

print("\n✓ All dependencies installed successfully!")
print("\nNext steps:")
print("1. Configure config/config.yaml with your API keys")
print("2. Run: python src/main.py")
EOF

python test_setup.py
```

---

## Configuration

### API Keys Setup

#### Unsplash API

1. Go to https://unsplash.com/oauth/applications
2. Click "New Application"
3. Accept terms and create app
4. Copy **Access Key** and **Secret Key**
5. Add to `config/config.yaml`:
```yaml
api_keys:
  unsplash:
    access_key: "YOUR_ACCESS_KEY"
    secret_key: "YOUR_SECRET_KEY"
```

**Rate Limits**: 
- Free tier: 50 requests/hour
- Sufficient for this project

#### Pexels API (Optional)

1. Go to https://www.pexels.com/api/
2. Sign up and verify email
3. Get your API key
4. Add to config:
```yaml
api_keys:
  pexels:
    api_key: "YOUR_PEXELS_KEY"
```

**Rate Limits**: 
- Free tier: 200 requests/hour

### Model Configuration

**For 6GB VRAM**:
```yaml
model:
  base_model: "runwayml/stable-diffusion-v1-5"
  device: "cuda"
  dtype: "float16"
  enable_attention_slicing: true
  enable_cpu_offload: false

generation:
  width: 512
  height: 512
  num_inference_steps: 25
```

**For 12GB+ VRAM**:
```yaml
model:
  base_model: "stabilityai/stable-diffusion-xl-base-1.0"
  device: "cuda"
  dtype: "float16"
  enable_attention_slicing: false

generation:
  width: 1024
  height: 1024
  num_inference_steps: 30
```

**For Apple Silicon**:
```yaml
model:
  base_model: "runwayml/stable-diffusion-v1-5"
  device: "mps"
  dtype: "float32"  # MPS works better with fp32
  enable_attention_slicing: true
```

---

## First Run

### Basic Test

```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run basic generation test
python src/test_generation.py
```

This should:
1. Load the model
2. Generate a test image
3. Save to `gallery/test/`

### First Artwork

```bash
# Generate your first AI artwork
python src/main.py --mode manual --prompt "a beautiful sunset"
```

### Start Automated Creation

```bash
# Start the scheduler (runs in background)
python src/main.py --mode auto

# Check logs
tail -f logs/ai_artist.log
```

---

## Troubleshooting

### Common Issues

#### "CUDA out of memory"
**Solution**:
```yaml
# In config.yaml
model:
  enable_attention_slicing: true
  enable_cpu_offload: true
generation:
  width: 512
  height: 512
```

#### "Module not found"
**Solution**:
```bash
# Verify virtual environment is activated
which python  # Should show venv path

# Reinstall requirements
pip install -r requirements.txt --upgrade
```

#### "API rate limit exceeded"
**Solution**:
- Check Unsplash dashboard for limits
- Enable Pexels as backup source
- Reduce creation frequency

#### "Model download failed"
**Solution**:
```bash
# Set Hugging Face token
export HF_TOKEN="your_token_here"

# Or use cached models
model:
  cache_dir: "models/cache"
```

#### macOS MPS Issues
**Solution**:
```yaml
model:
  device: "mps"
  dtype: "float32"  # Use fp32 instead of fp16
```

### Getting Help

1. Check logs: `logs/ai_artist.log`
2. Review [ARCHITECTURE.md](ARCHITECTURE.md)
3. Test each component individually
4. Check GitHub issues (if using version control)

---

## Next Steps

After successful setup:

1. **Review Configuration**: Check all settings in `config/config.yaml`
2. **Read Roadmap**: See [ROADMAP.md](ROADMAP.md) for development phases
3. **Start Development**: Begin with Phase 1 implementation
4. **Train Style**: Collect reference images for LoRA training
5. **Test Automation**: Run scheduler for 24 hours

---

## Updating

To update dependencies:

```bash
# Activate environment
source venv/bin/activate

# Update packages
pip install -r requirements.txt --upgrade

# Test after update
python test_setup.py
```

---

## Uninstallation

To completely remove:

```bash
# Remove virtual environment
rm -rf venv/

# Remove models (large files)
rm -rf models/cache/

# Remove generated content
rm -rf gallery/

# Keep source code and config
# Remove everything: rm -rf ~/Desktop/LizsDisk/ai-artist
```

---

## Performance Tips

1. **Use fp16**: 2x faster, half memory
2. **Enable attention slicing**: Reduces memory
3. **Batch size 1**: Most memory efficient
4. **Close other programs**: Free up VRAM
5. **Use SSD**: Faster model loading
6. **Monitor temps**: Keep GPU cool

---

## Quick Reference

**Activate environment**:
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**Run manual generation**:
```bash
python src/main.py --mode manual
```

**Start automation**:
```bash
python src/main.py --mode auto
```

**View gallery**:
```bash
python src/view_gallery.py
```

**Train LoRA** (Phase 2):
```bash
python src/train_lora.py --config config/training.yaml
```

You're now ready to build your autonomous AI artist! 🎨
