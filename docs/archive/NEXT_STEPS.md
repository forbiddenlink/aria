# Next Steps - What to Build

**Date:** January 9, 2026  
**Current Status:** Phase 5 Complete (Web Gallery) - Moving to Phase 6  
**Test Coverage:** 39% (52 tests passing)

---

## 🎯 Immediate Priorities (Next 2-3 Days)

### 1. Add Web Gallery Tests (HIGH PRIORITY)
**Goal**: Increase test coverage from 39% to 50%+

**Files to Create**:
```
tests/unit/test_web_api.py
tests/unit/test_websocket.py  
tests/unit/test_web_helpers.py
tests/unit/test_health.py
```

**What to Test**:
- [ ] **API Endpoints** (test_web_api.py)
  - `GET /api/images` with filters (limit, offset, featured, search)
  - `GET /api/images/file/{path}` - test security, path traversal prevention
  - `GET /api/stats` - verify calculations
  - `POST /api/generate` - test with valid/invalid inputs
  
- [ ] **WebSocket** (test_websocket.py)
  - Connection/disconnection
  - Message broadcasting
  - Session tracking
  - Multiple client handling
  
- [ ] **Helper Functions** (test_web_helpers.py)
  - `is_valid_image()` - test all validation rules
  - `load_image_metadata()` - test error handling
  - `filter_by_search()` - test case sensitivity
  - `calculate_gallery_stats()` - test accuracy

- [ ] **Health Checks** (test_health.py)
  - Basic health endpoint
  - Readiness probe
  - Liveness probe

**Expected Impact**: +11% coverage (web module currently 0%)

---

### 2. Complete Remaining Code TODOs

**File**: `src/ai_artist/curation/curator.py`

**TODO 1: Blur Detection** (Line 128)
```python
def _detect_blur(self, image: Image.Image) -> float:
    """Detect image blur using Laplacian variance.
    
    Returns:
        float: Blur score (0-1, higher is sharper)
    """
    import cv2
    import numpy as np
    
    # Convert PIL to numpy array
    img_array = np.array(image.convert('L'))  # Grayscale
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize: >100 is sharp, <50 is blurry
    blur_score = min(variance / 100.0, 1.0)
    
    logger.debug("blur_detection", variance=variance, score=blur_score)
    return blur_score
```

**TODO 2: Artifact Detection** (Line 128)
```python
def _detect_artifacts(self, image: Image.Image) -> float:
    """Detect compression artifacts and anomalies.
    
    Returns:
        float: Artifact score (0-1, higher is better)
    """
    import numpy as np
    
    img_array = np.array(image)
    
    # Check for extreme values (blown highlights/crushed shadows)
    extremes = np.sum((img_array < 10) | (img_array > 245))
    extreme_ratio = extremes / img_array.size
    
    # Check for color banding (posterization)
    unique_colors = len(np.unique(img_array))
    expected_colors = 256 * 3  # RGB
    color_diversity = unique_colors / expected_colors
    
    # Combine metrics (penalize extremes and low diversity)
    artifact_score = (1 - extreme_ratio) * 0.5 + color_diversity * 0.5
    
    logger.debug(
        "artifact_detection",
        extreme_ratio=extreme_ratio,
        color_diversity=color_diversity,
        score=artifact_score
    )
    return artifact_score
```

**TODO 3: Update Technical Score Calculation**
```python
def _calculate_technical_score(self, image: Image.Image) -> float:
    """Calculate technical quality score with blur and artifact detection."""
    # Resolution score
    width, height = image.size
    resolution = width * height
    resolution_score = min(resolution / (1024 * 1024), 1.0)
    
    # Blur detection
    blur_score = self._detect_blur(image)
    
    # Artifact detection  
    artifact_score = self._detect_artifacts(image)
    
    # Weighted combination
    technical_score = (
        resolution_score * 0.4 +
        blur_score * 0.4 +
        artifact_score * 0.2
    )
    
    logger.debug(
        "technical_score_calculated",
        resolution=resolution_score,
        blur=blur_score,
        artifacts=artifact_score,
        total=technical_score
    )
    
    return technical_score
```

**Dependencies Needed**:
```bash
pip install opencv-python
# Or add to requirements.txt:
opencv-python>=4.8.0
```

---

## 🚀 Phase 6: Deployment (Week 6 - Next Major Phase)

### 1. Docker Containerization

**Create Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p gallery models logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run web server
CMD ["uvicorn", "ai_artist.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Create docker-compose.yml**:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./gallery:/app/gallery
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: PostgreSQL for production
  # postgres:
  #   image: postgres:15
  #   environment:
  #     POSTGRES_DB: ai_artist
  #     POSTGRES_USER: ai_artist
  #     POSTGRES_PASSWORD: ${DB_PASSWORD}
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data

# volumes:
#   postgres_data:
```

**Create .dockerignore**:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv/
.git
.gitignore
.pytest_cache
htmlcov/
.coverage
*.log
.DS_Store
.history/
.claude/
*.md
!README.md
tests/
docs/
```

### 2. Deployment Platform Choices

**Option A: Railway.app (RECOMMENDED)**
- ✅ Easy deployment (connect GitHub repo)
- ✅ Free tier with $5 credit/month
- ✅ Automatic SSL certificates
- ✅ PostgreSQL add-on available
- ✅ Automatic deployments on push

**Setup**:
1. Create Railway account
2. Connect GitHub repo
3. Add environment variables (API keys)
4. Deploy!

**Option B: Render.com**
- ✅ Free tier available
- ✅ Auto-deploy from GitHub
- ✅ Managed PostgreSQL
- ⚠️ Slower than Railway

**Option C: DigitalOcean App Platform**
- ✅ $5/month
- ✅ Full control
- ✅ Scalable
- ⚠️ More configuration needed

**Option D: AWS/GCP/Azure**
- ✅ Enterprise-grade
- ✅ Unlimited scalability
- ⚠️ Complex setup
- ⚠️ Higher cost

### 3. CI/CD Enhancements

**Add Deployment Workflow** `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: |
          npm i -g @railway/cli
          railway up
```

---

## 📋 Phase 4 (Optional): Social Media Integration

**Only if user wants auto-posting to Instagram/Twitter**

### Instagram Integration
```python
# src/ai_artist/social/instagram.py
from instagrapi import Client

class InstagramPoster:
    def __init__(self, username: str, password: str):
        self.client = Client()
        self.client.login(username, password)
    
    def post_artwork(self, image_path: str, caption: str):
        """Post artwork to Instagram."""
        self.client.photo_upload(
            image_path,
            caption=caption
        )
```

### Twitter Integration
```python
# src/ai_artist/social/twitter.py
import tweepy

class TwitterPoster:
    def __init__(self, api_key: str, api_secret: str, 
                 access_token: str, access_secret: str):
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_secret)
        self.api = tweepy.API(auth)
    
    def post_artwork(self, image_path: str, caption: str):
        """Post artwork to Twitter."""
        self.api.update_status_with_media(
            caption,
            image_path
        )
```

---

## 🎨 Future Enhancements (Phase 7+)

### Advanced Features to Consider
1. **Multi-Model Support**
   - SDXL, SD 2.1, Midjourney-style models
   - Model switching via API

2. **Advanced Curation**
   - BRISQUE quality assessment
   - Multi-metric scoring
   - Image hash deduplication

3. **User Features**
   - User accounts and authentication
   - Personal galleries
   - Favorites system
   - Download collections as ZIP

4. **Generation Enhancements**
   - ControlNet support (preserve structure)
   - Img2Img pipeline
   - Inpainting
   - Upscaling (Real-ESRGAN)

5. **Analytics Dashboard**
   - Generation statistics
   - Popular prompts
   - Quality trends over time
   - Cost tracking

6. **Mobile App**
   - React Native or Flutter
   - Push notifications for completed generations
   - Gallery browsing
   - On-device generation (maybe)

---

## 📊 Success Metrics

### Short-term Goals (This Week)
- [ ] Web gallery tests complete
- [ ] Coverage increased to 50%+
- [ ] Blur/artifact detection implemented
- [ ] Docker setup complete

### Medium-term Goals (Next 2 Weeks)
- [ ] Deployed to production (Railway/Render)
- [ ] Custom domain configured
- [ ] CI/CD fully automated
- [ ] Basic monitoring in place

### Long-term Goals (Next Month)
- [ ] 60%+ test coverage
- [ ] Social media integration (optional)
- [ ] Advanced curation features
- [ ] Performance optimizations

---

## 🎯 Recommended Next Actions

**Today**:
1. ✅ Documentation updated
2. ✅ Redundant files removed
3. Start web gallery tests

**Tomorrow**:
1. Complete web gallery tests
2. Implement blur/artifact detection
3. Test coverage report

**Next Week**:
1. Create Dockerfile
2. Set up docker-compose
3. Choose deployment platform
4. Deploy to production!

---

**Questions to Consider**:
- Do you want social media auto-posting? (Phase 4)
- Which deployment platform do you prefer?
- Do you need PostgreSQL or is SQLite sufficient?
- Should we add user authentication?
- Any specific features you want prioritized?

---

Ready to start building! 🚀
