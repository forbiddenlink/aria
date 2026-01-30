# 🚀 Quick Start Guide

Get up and running with AI Artist in 10 minutes!

---

## 📋 Prerequisites

Before you begin, make sure you have:

- **Python 3.11+** installed
- **8GB+ RAM** (16GB recommended)
- **10GB+ free disk space**
- **GPU** (optional but recommended):
  - NVIDIA GPU with CUDA support, OR
  - Apple Silicon (M1/M2/M3) with MPS support

---

## 🎯 Installation (3 Steps)

### Step 1: Clone and Navigate

```bash
git clone https://github.com/yourusername/ai-artist.git
cd ai-artist
```

### Step 2: Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure API Keys

```bash
# Copy the example config
cp config/config.example.yaml config/config.yaml

# Edit with your favorite editor
nano config/config.yaml
```

**Add your Unsplash API key:**

1. Go to https://unsplash.com/developers
2. Create an app
3. Copy your Access Key
4. Paste it in `config.yaml`:

```yaml
api_keys:
  unsplash_access_key: "YOUR_KEY_HERE"
```

---

## 🎨 Your First Artwork (30 seconds)

Generate your first AI artwork:

```bash
ai-artist --theme "sunset over mountains"
```

That's it! Your image will be saved to `gallery/YYYY/MM/DD/`.

---

## 🖼️ View Your Gallery

Launch the web interface:

```bash
ai-artist-web
```

Then open your browser to: http://localhost:8000

---

## 📅 Automated Daily Art

Want AI Artist to create art automatically?

```bash
# Create art daily at 9 AM
ai-artist-schedule start daily --hour 9

# Check schedule status
ai-artist-schedule status
```

---

## 🎭 Train Your Own Style (Optional)

Create a unique artistic style with LoRA:

```bash
# 1. Prepare 15-30 images in datasets/training/
# 2. Run training
python src/ai_artist/training/train_lora.py \
  --instance_data_dir datasets/training \
  --output_dir models/lora/my_style \
  --max_train_steps 2000

# 3. Update config to use your LoRA
# Edit config.yaml:
#   lora_path: "models/lora/my_style"
```

Training takes 20-40 minutes on Apple Silicon.

---

## 🆘 Common Issues

### "No module named 'ai_artist'"

Install in editable mode:

```bash
pip install -e .
```

### "CUDA out of memory"

Use CPU or reduce image size:

```yaml
# In config.yaml
model:
  device: "cpu"  # or "mps" for Apple Silicon
generation:
  width: 768
  height: 768
```

### "API rate limit exceeded"

Unsplash free tier: 50 requests/hour. Wait or upgrade.

### Images are black/corrupted (MPS)

This is a known issue with MPS. The code includes fixes:

```python
# Already implemented in generator.py
# VAE uses float32 on MPS
```

If issues persist, try:

```yaml
model:
  dtype: "float32"  # Instead of float16
```

---

## 📚 Next Steps

**Learn more:**
- 📖 [Setup Guide](SETUP.md) - Detailed installation
- 🏗️ [Architecture](ARCHITECTURE.md) - How it works
- 🎨 [LoRA Training](LORA_TRAINING.md) - Advanced styling
- 🌐 [Web Gallery](docs/WEB_GALLERY.md) - UI features
- 🔒 [Security](SECURITY.md) - Best practices

**Get creative:**
- Try different themes and prompts
- Experiment with generation settings
- Train a LoRA on your own art style
- Set up automated creation schedules

**Join the community:**
- ⭐ Star the repo if you like it!
- 🐛 Report bugs via GitHub Issues
- 💡 Suggest features
- 🤝 Contribute improvements

---

## 🎉 You're All Set!

Your AI Artist is ready to create. Happy generating! 🎨

**Quick Commands Reference:**

```bash
# Generate single image
ai-artist --theme "your prompt here"

# Launch web gallery
ai-artist-web

# View gallery in terminal
ai-artist-gallery

# Start daily automation
ai-artist-schedule start daily

# Check logs
tail -f logs/ai_artist.log
```

---

**Need help?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue on GitHub.
