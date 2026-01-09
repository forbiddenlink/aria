# Quick Start Guide - AI Artist

## Installation Complete! ✅

Your AI Artist is fully set up and ready to create amazing art!

## Usage Commands

### Generate Art

```bash
# Generate with random theme
ai-artist

# Generate with specific theme
ai-artist --theme "sunset landscape"
ai-artist --theme "cyberpunk city"
ai-artist --theme "abstract watercolor"
ai-artist --theme "peaceful zen garden"

# Run automated mode (daily at 9 AM)
ai-artist --mode auto
```

### View Your Gallery

```bash
# List recent images
ai-artist-gallery list

# List more images
ai-artist-gallery list --limit 20

# Open an image
ai-artist-gallery open gallery/2026/01/09/image.png
```

## What's New? 🎉

### 1. **Smart Curation System** ⭐
- Generates **3 variations** of each image
- Uses CLIP to evaluate quality (aesthetic score, technical quality, prompt alignment)
- Automatically saves the **best image**
- Shows quality scores for each variation

### 2. **Enhanced Prompt Engineering** 🎨
- Random artistic style modifiers
- More creative and varied prompts
- Better quality outputs

Styles include:
- "masterpiece, highly detailed, professional photography"
- "artistic interpretation, vivid colors, detailed composition"
- "beautiful artwork, intricate details, stunning visual"
- "creative composition, high quality, aesthetically pleasing"

### 3. **Progress Indicators** 📊
- Real-time progress bar during generation
- Clear feedback at each step
- See exactly what's happening

### 4. **Gallery Viewer** 🖼️
- Browse all generated images
- See file sizes and dimensions
- Quick access to recent creations

### 5. **Git Repository** 📝
- Version control initialized
- Proper .gitignore configured
- Ready for collaboration

## Example Output

When you run `ai-artist --theme "mountain sunset"`, you'll see:

```
🎨 Generating 3 variations...
   [████████████████████░░░░░░] 67% (20/30 steps)

🔍 Evaluating image quality...
   Image 1: score=0.723 (aesthetic=0.65, clip=0.82)
   Image 2: score=0.756 (aesthetic=0.71, clip=0.79)
   Image 3: score=0.689 (aesthetic=0.62, clip=0.75)

✨ Selected best image with score: 0.756
```

## Generated Images Location

```
gallery/
├── 2026/
│   └── 01/
│       └── 09/
│           ├── 20260109_103000_noseed.png
│           ├── 20260109_102800_noseed.png
│           └── archive/
```

## Configuration

Edit `config/config.yaml` to customize:
- Model settings (different Stable Diffusion versions)
- Image dimensions
- Number of variations
- Negative prompts
- API keys

## Next Steps

### Quick Improvements
1. **Test the new features** - Generate some art!
2. **Explore different themes** - Try various prompts
3. **Review quality scores** - See how curation works

### Medium-term Enhancements (from ROADMAP.md)
1. **LoRA Style Training** - Create your unique artistic style
2. **Social Media Integration** - Auto-post to Instagram/Twitter
3. **Advanced Scheduling** - Multiple daily generations
4. **Database Tracking** - Store metadata and analytics

### Foundation Tasks
1. ✅ Git repository initialized
2. ⏳ Pre-commit hooks (run tests automatically)
3. ⏳ Expanded test coverage (integration & E2E tests)
4. ⏳ Structured logging with `structlog`

## Troubleshooting

### Black Images
Fixed! Using Stable Diffusion 1.5 with float32 on MPS.

### Slow Generation
Normal on Apple Silicon MPS (~40 seconds per image). GPU is working!

### Out of Memory
Reduce `num_variations` in config from 3 to 1 or 2.

## Support

- Check [ROADMAP.md](ROADMAP.md) for planned features
- See [IMPROVEMENTS.md](IMPROVEMENTS.md) for enhancement ideas
- Review [TESTING.md](TESTING.md) for quality assurance

---

**Your AI Artist is ready to create! 🎨✨**

Run `ai-artist` to generate your first curated artwork!
