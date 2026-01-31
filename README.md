# Aria - Autonomous AI Artist

An autonomous AI artist with personality, moods, memory, and creative independence.

## What Makes Aria Different

Aria isn't a tool - she's an **artist** you collaborate with:

- **Has Moods**: 10 emotional states that influence her art
- **Remembers**: Episodic + semantic memory learns what works
- **Reflects**: Journals her thoughts after each creation
- **Chooses**: Makes autonomous decisions about what to paint
- **Evolves**: Artistic style develops through experience
- **Ethical AI**: Built-in bias mitigation to ensure fair and diverse output

## Quick Start

```bash
# Clone and setup
cd ai-artist
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp config/config.example.yaml config/config.yaml

# Let Aria create (she chooses based on mood)
python -m ai_artist.main

# Or suggest a theme
python -m ai_artist.main --theme "twilight dreams"

# Web gallery
python -m ai_artist.web.app
```

## Project Structure

```
ai-artist/
├── src/ai_artist/
│   ├── personality/    # Moods, memory, identity
│   ├── core/           # Image generation
│   ├── curation/       # CLIP-based quality scoring
│   ├── scheduling/     # Autonomous creation
│   └── web/            # FastAPI gallery
├── config/             # Configuration
├── gallery/            # Generated artwork
└── docs/               # Documentation
```

## Documentation

| Document | Purpose |
|----------|---------|
| **[ARIA.md](ARIA.md)** | Full personality system & roadmap |
| **[QUICKSTART.md](QUICKSTART.md)** | Step-by-step setup guide |
| **[SETUP.md](SETUP.md)** | Detailed installation |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues |
| **[LORA_GUIDE.md](LORA_GUIDE.md)** | Custom style training |
| **[docs/](docs/)** | Technical documentation |

## Current Status

**Implemented:**

- 10-mood personality system
- 3-layer memory (episodic/semantic/working)
- Autonomous scheduling
- CLIP-based quality curation
- FastAPI web gallery with WebSocket
- Bias mitigation in prompt generation
- Intelligent prompt filtering
- **Advanced Prompt Features**: Matrix, Emphasis, Style Presets

**In Progress:**

- Critique system for iterative improvement
- Visible thinking process
- Multi-model support
- Beautiful dark UI

See [ARIA.md](ARIA.md) for the complete roadmap and
[docs/ADVANCED_PROMPTS.md](docs/ADVANCED_PROMPTS.md) for prompt utilities.

## Tech Stack

- Stable Diffusion (diffusers)
- LoRA for style training
- FastAPI + WebSocket
- SQLite
- CLIP for curation

## License

MIT License
