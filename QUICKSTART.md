# Quick Start Guide

## 🎯 Overview

This guide helps you get started with the AI Artist project after the comprehensive improvements.

---

## 📚 Documentation Map

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [README.md](README.md) | Project overview & features | Start here |
| [SETUP.md](SETUP.md) | Installation guide | Before coding |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design | Understanding structure |
| [ROADMAP.md](ROADMAP.md) | Development timeline | Planning work |
| [LEGAL.md](LEGAL.md) | **Copyright & compliance** | **Before training models** |
| [TESTING.md](TESTING.md) | Testing strategy | Writing tests |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev workflow | Before contributing |
| [SECURITY.md](SECURITY.md) | Security practices | Handling secrets |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | What changed & why | Understanding updates |

---

## 🚀 Getting Started (Phase 0.5)

### Step 1: Prerequisites (5 minutes)

```bash
# Install Python 3.10+
python3 --version  # Should be 3.10 or higher

# Check Git
git --version

# Get API keys
# 1. Unsplash: https://unsplash.com/developers
# 2. Pexels (optional): https://www.pexels.com/api/
```

### Step 2: Project Setup (10 minutes)

```bash
# Navigate to project
cd /Volumes/LizsDisk/ai-artist

# Initialize Git (if not already done)
git init
git add .
git commit -m "feat: initial project setup with comprehensive documentation"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configuration (5 minutes)

```bash
# Create .env file for secrets
cat > .env << 'EOF'
UNSPLASH_ACCESS_KEY=your_access_key_here
UNSPLASH_SECRET_KEY=your_secret_key_here
PEXELS_API_KEY=your_pexels_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
EOF

# Copy example config
cp config/config.example.yaml config/config.yaml

# Edit config with your preferences
# nano config/config.yaml
# or
# code config/config.yaml
```

### Step 4: Development Tools (5 minutes)

```bash
# Install pre-commit hooks
pre-commit install

# Verify installation
pytest --version
black --version
ruff --version

# Run initial tests (will create test structure)
pytest tests/ || echo "Tests directory not yet created"
```

### Step 5: Verify Setup (5 minutes)

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test imports
python -c "import diffusers, transformers, PIL; print('✅ All imports successful')"

# Run security check
safety check || pip-audit
```

---

## 📋 Phase 0.5 Checklist

Use this checklist to complete the foundation phase:

### Git & Version Control
- [ ] Git repository initialized
- [ ] First commit created
- [ ] `.gitignore` reviewed
- [ ] Remote repository set up (optional)

### Development Environment
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] PyTorch working (GPU detection verified)
- [ ] Pre-commit hooks installed

### Configuration & Secrets
- [ ] `.env` file created with API keys
- [ ] `config.yaml` customized
- [ ] Secrets not committed to git (verified)
- [ ] API keys tested with simple request

### Code Quality
- [ ] Black formatter configured
- [ ] Ruff linter configured
- [ ] Pre-commit running on commits
- [ ] Sample test file created

### Testing Framework
- [ ] pytest installed and working
- [ ] Test directory structure created
- [ ] Sample unit test written
- [ ] Coverage tool configured

### Documentation Review
- [ ] **LEGAL.md** read and understood
- [ ] SECURITY.md guidelines noted
- [ ] TESTING.md strategy reviewed
- [ ] CONTRIBUTING.md workflow understood

### Legal Compliance
- [ ] Training data strategy defined
- [ ] Public domain sources identified
- [ ] Copyright considerations documented
- [ ] Artist blocking list prepared (if needed)

---

## 🎨 Phase 1: First Generation

### Week 1: Infrastructure

```bash
# Create project structure
mkdir -p src/{inspiration,generation,training,curation,gallery,scheduling}
mkdir -p tests/{unit,integration,e2e,fixtures}
mkdir -p {models/lora,gallery,data,logs}

# Create __init__.py files
touch src/{__init__.py,inspiration,generation,training,curation,gallery,scheduling}/__init__.py

# Create first module with error handling
cat > src/inspiration/unsplash_client.py << 'EOF'
"""Unsplash API client with retry logic."""
from typing import Dict, Optional
import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

class UnsplashClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("UNSPLASH_ACCESS_KEY")
        self.base_url = "https://api.unsplash.com"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
    def fetch_random_image(self, query: Optional[str] = None) -> Dict:
        """Fetch random image with automatic retry."""
        # Implementation here
        pass
EOF

# Create first test
cat > tests/unit/test_unsplash_client.py << 'EOF'
"""Tests for Unsplash client."""
import pytest
from src.inspiration.unsplash_client import UnsplashClient

def test_client_initialization():
    client = UnsplashClient(api_key="test_key")
    assert client.api_key == "test_key"

# More tests...
EOF

# Run tests
pytest tests/unit/test_unsplash_client.py
```

### Week 2: Core Generation

Follow the detailed steps in [ROADMAP.md](ROADMAP.md) Phase 1, Week 2.

---

## 🔍 Common Commands

### Development

```bash
# Activate environment
source venv/bin/activate

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Check security
safety check
pip-audit
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_inspiration.py

# Run with verbose output
pytest -v

# Run and stop at first failure
pytest -x

# Run only fast tests
pytest -m "not slow"
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/unsplash-client

# Make changes, then:
git add .
git commit -m "feat: add Unsplash API client with retry logic"

# Push to remote
git push origin feature/unsplash-client
```

---

## ⚠️ Important Reminders

### Before You Start Coding

1. ✅ Read **LEGAL.md** - Copyright is critical!
2. ✅ Set up secrets properly - Never commit API keys
3. ✅ Configure pre-commit hooks - Catch issues early
4. ✅ Write tests first - TDD when possible

### During Development

1. 📝 Log everything with structlog
2. 🔄 Handle errors with retry logic
3. 🧪 Write tests for new features
4. 📚 Document as you go
5. 🔒 Keep secrets in .env

### Before Committing

1. ✅ Run tests (`pytest`)
2. ✅ Check coverage (`pytest --cov`)
3. ✅ Format code (`black src/`)
4. ✅ Lint code (`ruff check src/`)
5. ✅ Review changes (`git diff`)

---

## 🆘 Troubleshooting

### "Module not found" errors

```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "CUDA not available"

```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "API rate limit exceeded"

- Check your API usage on provider dashboard
- Implement rate limiting in your code
- Consider adding Pexels as fallback

### Tests failing

```bash
# Run with verbose output to see details
pytest -v

# Run single test to isolate issue
pytest tests/unit/test_file.py::test_function -v

# Check if it's a dependency issue
pip install -r requirements.txt --upgrade
```

---

## 📊 Success Metrics

Track these to measure progress:

- [ ] All Phase 0.5 checklist items complete
- [ ] At least 3 passing unit tests
- [ ] Pre-commit hooks running automatically
- [ ] Code formatted with Black
- [ ] No linting errors from Ruff
- [ ] Security scan passes (safety/pip-audit)
- [ ] Documentation reviewed
- [ ] Ready to start Phase 1

---

## 🎯 Next Steps

After completing this quick start:

1. **Week 1**: Implement Inspiration Engine (see [ROADMAP.md](ROADMAP.md))
2. **Week 2**: Build Generation Pipeline
3. **Weeks 3-4**: Train LoRA style
4. **Week 5**: Set up automation
5. **Weeks 6-8**: Add advanced features

---

## 📞 Getting Help

If you encounter issues:

1. Check relevant documentation (see Document Map above)
2. Review [SETUP.md](SETUP.md) troubleshooting section
3. Check test output for specific errors
4. Review logs in `logs/` directory
5. Consult [CONTRIBUTING.md](CONTRIBUTING.md) for development help

---

## ✅ You're Ready!

Once you've completed Phase 0.5, you have:

- ✅ Solid development foundation
- ✅ All tools configured
- ✅ Security properly set up
- ✅ Legal guidelines understood
- ✅ Testing framework ready
- ✅ Code quality tools working
- ✅ Ready to build features!

**Time to create some art! 🎨**

---

*Quick Start Guide | Version 2.0 | Last Updated: January 8, 2026*
