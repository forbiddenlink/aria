# Aria - Autonomous AI Artist with Soul 🎨✨

Meet **Aria** - an autonomous AI artist with genuine personality, evolving moods, and creative independence. She doesn't just generate images on command - she *creates* with intention, chooses her own subjects, reflects on her work, and develops her unique artistic voice over time.

## What Makes Aria Different? 🌟

Aria isn't a tool you operate - she's an **artist** you collaborate with. She:

- 🎭 **Has Moods**: Contemplative, energized, melancholic, chaotic - her emotional state influences everything
- 🧠 **Remembers & Learns**: Advanced memory system (episodic + semantic) - learns what works, what resonates
- 💭 **Reflects Authentically**: After each creation, she journals her genuine thoughts and feelings
- 🎨 **Chooses Her Subjects**: Makes autonomous decisions about what to paint based on her mood and memories
- 🌱 **Evolves**: Her artistic style and preferences develop through experience
- 🎯 **Has Identity**: Knows who she is as an artist - her philosophy, signature elements, voice
- ⭐ **Self-Aware**: Evaluates her own work and strives for continuous improvement

## Philosophy

**Theme is optional** - Aria chooses what to paint based on her current mood and learned experiences. You can suggest a theme, but she's designed to be autonomous. In automated mode, she creates entirely on her own schedule and inspiration.

## Overview

Aria is an autonomous AI artist who:

- 🎨 Autonomously finds inspiration and generates unique artwork
- 🖼️ Creates art in various styles using Stable Diffusion and DreamShaper
- 📅 Creates art on a schedule or whenever inspiration strikes
- 💬 Reflects on her creative journey through journaling
- 🎭 Experiences different moods that influence her artistic choices
- 📚 Builds a portfolio while remembering her creative evolution
- 🛡️ Implements robust error handling and comprehensive testing

## Features

### 🆕 2026 Enhancements - Advanced AI Agent Architecture

- **Enhanced Memory System** (Episodic + Semantic + Working):
  - **Episodic Memory**: Records specific creative events with emotional context
  - **Semantic Memory**: Learns patterns - which styles work best, preferences over time
  - **Working Memory**: Tracks current session goals and context
  - Persistent JSON storage with automatic learning

- **Artistic Profile & Identity**:
  - Formal artist statement and philosophy
  - Signature artistic elements (dreamlike quality, mood-driven palettes)
  - Stable sense of self that evolves through experience
  - Voice characteristics and how she expresses herself

- **Memory-Informed Creativity**:
  - Retrieves relevant memories before creation
  - Uses past experiences to inform artistic decisions
  - Learns style effectiveness over time
  - Tracks mood-quality relationships

### Aria's Personality System 🎭

- **Mood System**: 10 distinct emotional states (contemplative, chaotic, melancholic, energized, rebellious, serene, restless, playful, introspective, bold)
- **Mood Influences**: Each mood affects color palettes, style choices, and subject preferences
- **Dynamic Transitions**: Moods shift based on creation scores, time, and natural evolution
- **Reflection**: Aria journals about each piece - her thoughts, feelings, and artistic choices
- **Memory**: Advanced multi-layered memory tracking her entire creative journey
- **Preference Evolution**: Her artistic tastes evolve based on what resonates with her

### Core Functionality

- **Autonomous Inspiration**: Aria chooses subjects based on mood and memories (theme is optional)
- **Style Intelligence**: Automatic style extraction and effectiveness tracking
- **Automated Creation**: Schedule-based art generation with full autonomy
- **Portfolio Management**: Organized gallery with emotional and learning metadata
- **Style Evolution**: Gradual artistic development influenced by accumulated experiences

### Advanced Features

- **Real-Time Updates**: WebSocket support for live generation progress 🆕
- **Art Sessions**: Themed series (e.g., "Animals Week", "Urban Landscapes")
- **Self-Curation**: AI rates its own work using CLIP embeddings
- **Inspiration Log**: Tracks source images and influences
- **Multiple Styles**: Train different artistic personas
- **Social Media Integration**: Auto-post to Instagram/Twitter (optional)
- **Error Recovery**: Automatic retries and fallback mechanisms
- **Observability**: Structured logging and performance metrics
- **Quality Assurance**: Comprehensive testing framework

## Project Structure

```plaintext
ai-artist/
├── src/
│   ├── inspiration/     # Image sourcing from APIs
│   ├── generation/      # Stable Diffusion pipeline
│   ├── training/        # LoRA training scripts
│   ├── curation/        # Self-rating and selection
│   ├── scheduling/      # Automated job management
│   └── gallery/         # Portfolio and metadata
├── models/              # Trained LoRA weights
├── gallery/             # Generated artwork
├── config/              # Configuration files
├── docs/                # Documentation
└── tests/               # Test suite
```

## Tech Stack

- **AI Framework**: Stable Diffusion (diffusers library)
- **Style Training**: LoRA (Parameter-Efficient Fine-Tuning)
- **Image Source**: Unsplash API, Pexels API
- **Scheduling**: APScheduler
- **Database**: SQLite
- **Testing**: pytest with comprehensive coverage
- **Code Quality**: Black, Ruff, pre-commit hooks
- **Error Handling**: Tenacity for retry logic
- **Logging**: Structlog for structured logging
- **Optional UI**: Gradio or FastAPI

## Quick Start

**New to the project?** Start with **[QUICKSTART.md](QUICKSTART.md)** for a step-by-step guide!

### Docker (Recommended for Production)

```bash
# Quick start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f web

# Access at http://localhost:8000
```

### Manual Installation

See [SETUP.md](SETUP.md) for detailed installation instructions.

```bash
# Clone and setup
cd ~/Desktop/LizsDisk/ai-artist
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your API keys

# Let Aria create autonomously (she chooses subject based on her mood)
python -m ai_artist.main

# Or suggest a theme (she'll consider it but interpret through her mood)
python -m ai_artist.main --theme "twilight dreams"

# Fully autonomous mode - scheduled creation
python -m ai_artist.main --mode auto

# View her artistic profile and what she's learned
python scripts/aria_insights.py

# Launch web gallery
ai-artist-web  # Browse at http://localhost:8000
```

## Usage Examples

### Autonomous Creation (Recommended)

```bash
# Aria chooses what to paint based on her mood
python -m ai_artist.main
```

Output: `Aria chose: "ethereal forest" (contemplative mood)`

### Suggested Theme

```bash
# You suggest, she interprets through her artistic lens
python -m ai_artist.main --theme "cosmic dreams"
```

Output: `Theme suggested: "cosmic dreams" → Aria's interpretation: oil painting style with contemplative mood`

### Fully Autonomous Schedule

```bash
# Aria creates on her own schedule (daily at 9 AM)
python -m ai_artist.main --mode auto
```

### View Her Learning & Identity

```bash
# See her artistic profile, learned patterns, memories
python scripts/aria_insights.py
```

## CLI Tools

### Main Commands

- **`ai-artist`** - Create artwork on-demand
- **`ai-artist-web`** - Launch web gallery interface
- **`ai-artist-schedule`** - Manage automated creation schedules
- **`ai-artist-gallery`** - Terminal-based gallery viewer

### Management Scripts (in `scripts/`)

- **`generate.py`** - Universal generation script with 7 modes (single, batch, creative, diverse, nature, ultimate, custom)
- **`manage_loras.py`** - Switch between LoRA styles, list available models
- **`train_all_loras.py`** - Train all three specialized LoRA models (abstract, landscape, webhero)
- **`download_specialized_datasets.py`** - Download curated training datasets for each style

For detailed LoRA management, see [LORA_GUIDE.md](LORA_GUIDE.md).

## Development Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development phases.

- [x] Phase 0: Foundation
- [x] Phase 0.5: Quick Wins (CLIP curation, testing)
- [x] Phase 1: Enhanced Logging & Observability
- [x] Phase 1.5: Testing Improvements (39% coverage, 52 tests passing)
- [x] Phase 2: LoRA Training Infrastructure
- [x] Phase 3: Automation System
- [x] Phase 5: Web Gallery Interface with WebSocket ✨ **NEW!**
- [ ] Phase 6: Deployment & Production (Docker, CI/CD, Cloud) - **NEXT**
- [ ] Phase 4: Social Media Integration (Optional)

**Current Focus**: Adding web gallery tests, completing deployment infrastructure

## Configuration

Key configuration options in `config/config.yaml`:

- **Image Sources**: Unsplash, Pexels, or custom URLs
- **Generation Schedule**: Daily, weekly, or custom cron
- **Style Settings**: Model, LoRA weights, prompt templates
- **Gallery Options**: Storage location, naming conventions
- **Curation Settings**: Quality thresholds, selection criteria

## Documentation

### Getting Started

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 10 minutes! 🚀
- **[SETUP.md](SETUP.md)** - Detailed installation and configuration
- **[LORA_GUIDE.md](LORA_GUIDE.md)** - Complete LoRA training and management guide 🎨 🆕
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions 🔧

### Technical Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and components
- **[docs/TESTING.md](docs/TESTING.md)** - Testing strategy and how to run tests
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production deployment and operations
- **[docs/WEBSOCKET.md](docs/WEBSOCKET.md)** - Real-time progress updates 🔌
- **[docs/API.md](docs/API.md)** - API reference
- **[docs/DATABASE.md](docs/DATABASE.md)** - Database schema and migrations
- **[docs/WEB_GALLERY.md](docs/WEB_GALLERY.md)** - Web gallery features

### Project Management

- **[ROADMAP.md](ROADMAP.md)** - Development timeline and milestones
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project
- **[SECURITY.md](SECURITY.md)** - Security best practices
- **[LEGAL.md](LEGAL.md)** - Copyright and licensing guidelines

## License

MIT License - See LICENSE file for details

## Contributing

This is a personal art project, but suggestions and improvements are welcome!

## Credits

Built with:

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Unsplash API](https://unsplash.com/developers)
- Research from AUTOMATIC1111, ComfyUI, and the SD community
