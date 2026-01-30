# AI Artist - Autonomous Art Generator

An autonomous AI artist that discovers inspiration online and creates unique artwork in its own signature style.

## Overview

This project creates an AI that:

- 🎨 Autonomously finds inspiration from image APIs (Unsplash, Pexels)
- 🖼️ Generates artwork in a consistent, unique style using Stable Diffusion
- 📅 Creates art on a schedule (daily, weekly, or custom)
- 📚 Builds a growing portfolio over time
- 🎭 Evolves its artistic style gradually
- ⭐ Curates its own work, showcasing favorites
- 🔒 Follows legal and copyright best practices
- 🛡️ Implements robust error handling and monitoring

## Features

### Core Functionality

- **Autonomous Inspiration**: Pulls random images from Unsplash API
- **Style Consistency**: Uses LoRA fine-tuning for unique artistic voice
- **Automated Creation**: Schedule-based art generation
- **Portfolio Management**: Organized gallery with metadata
- **Style Evolution**: Gradual artistic development over time

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

# Generate artwork
python src/main.py

# Launch web gallery
ai-artist-web  # Browse at http://localhost:8000

# Schedule automated creation
ai-artist-schedule start daily
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
