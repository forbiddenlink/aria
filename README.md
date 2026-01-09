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
- **Art Sessions**: Themed series (e.g., "Animals Week", "Urban Landscapes")
- **Self-Curation**: AI rates its own work using CLIP embeddings
- **Inspiration Log**: Tracks source images and influences
- **Multiple Styles**: Train different artistic personas
- **Social Media Integration**: Auto-post to Instagram/Twitter (optional)
- **Error Recovery**: Automatic retries and fallback mechanisms
- **Observability**: Structured logging and performance metrics
- **Quality Assurance**: Comprehensive testing framework

## Project Structure

```
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

- **`ai-artist`** - Create artwork on-demand
- **`ai-artist-web`** - Launch web gallery interface
- **`ai-artist-schedule`** - Manage automated creation schedules
- **`ai-artist-gallery`** - Terminal-based gallery viewer

## Development Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development phases.

- [x] Phase 0: Foundation
- [x] Phase 0.5: Quick Wins (CLIP curation, testing)
- [x] Phase 1: Enhanced Logging & Observability
- [x] Phase 1.5: Testing Improvements (58% coverage)
- [x] Phase 2: LoRA Training Infrastructure
- [x] Phase 3: Automation System
- [x] Phase 5: Web Gallery Interface ✨ NEW!
- [ ] Phase 6: Deployment & Production

## Configuration

Key configuration options in `config/config.yaml`:

- **Image Sources**: Unsplash, Pexels, or custom URLs
- **Generation Schedule**: Daily, weekly, or custom cron
- **Style Settings**: Model, LoRA weights, prompt templates
- **Gallery Options**: Storage location, naming conventions
- **Curation Settings**: Quality thresholds, selection criteria

## Documentation

- **[SETUP.md](SETUP.md)** - Installation and configuration guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and components
- **[ROADMAP.md](ROADMAP.md)** - Development timeline and milestones
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment and operations
- **[COMPLIANCE.md](COMPLIANCE.md)** - GDPR, EU AI Act, and ethical AI
- **[LEGAL.md](LEGAL.md)** - Copyright and licensing guidelines
- **[TESTING.md](TESTING.md)** - Testing strategy and how to run tests
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project
- **[SECURITY.md](SECURITY.md)** - Security best practices

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
