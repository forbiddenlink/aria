#!/bin/bash
# Setup script for AI Artist project

set -e

echo "🎨 Setting up AI Artist project..."

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Installing from pyproject.toml..."
    pip install -e ".[dev]"
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p models/{lora,cache}
mkdir -p gallery
mkdir -p data
mkdir -p logs
mkdir -p config
mkdir -p datasets/{training,regularization}

# Copy example config if config doesn't exist
if [ ! -f "config/config.yaml" ]; then
    if [ -f "config/config.example.yaml" ]; then
        echo "Creating config.yaml from example..."
        cp config/config.example.yaml config/config.yaml
        echo "⚠️  Please edit config/config.yaml with your API keys"
    fi
fi

# Initialize database
echo "Initializing database..."
mkdir -p data
alembic upgrade head || echo "⚠️  Database migration failed - will create on first run"

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit hooks..."
    pre-commit install
else
    echo "⚠️  pre-commit not found, skipping hooks setup"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config/config.yaml with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Test: python -m pytest tests/"
echo "4. Generate art: python -m src.ai_artist.main --mode manual"
echo ""
echo "For more information, see QUICKSTART.md"

