# AI Artist Improvements Handoff

**Date**: 2026-01-30
**Session**: Codebase review and improvements

## Completed Work

### 1. Black/Blank Image Detection & Cleanup
- **Added** `is_black_or_blank()` function in `src/ai_artist/curation/curator.py`
- **Updated** `ImageGenerator.generate()` to filter invalid images before returning
- **Added** `GalleryManager.cleanup_invalid_images()` method
- **Created** `scripts/cleanup_gallery.py` CLI tool
- **Added** `POST /api/gallery/cleanup` endpoint
- **Deleted** 30 black images from gallery (649 valid remain)

### 2. MPS Stability Fix
- Changed `config/config.yaml` dtype from `float16` to `float32`
- Added warning in generator when MPS + float16 detected

### 3. Concurrency Controls
- Added semaphore (max 1 concurrent generation) in `app.py`
- Added 5-minute timeout for generations
- Added queue status to `/health` endpoint

### 4. Deployment Configuration
- **Vercel**: Created `vercel.json` and `api/index.py` (gallery-only mode)
- **Docker**: Improved `Dockerfile` with multi-stage build, non-root user
- **GPU Docker**: Created `Dockerfile.gpu` and `docker-compose.gpu.yml`

### 5. Face Quality Improvements
- Enhanced negative prompt with face-specific terms
- Created `config/wildcards/quality.txt` and `face_quality.txt`
- Created `src/ai_artist/core/face_restore.py` (GFPGAN integration)
- Added `face_restoration` config section

## Not Yet Implemented

### Face Restoration Integration
The `face_restore.py` module is created but not yet wired into the main pipeline.
To complete:
1. Update `src/ai_artist/main.py` to use `FaceRestorer` after generation
2. Add `face_restoration` config parsing in `utils/config.py`
3. Test with: `pip install gfpgan` then generate

### Better Model Switch
For consistently better faces, user could switch to:
- `Lykon/dreamshaper-8` - Great faces, SD 1.5 compatible
- `Lykon/realistic-vision-v5-1` - Photorealistic
- SDXL (needs more VRAM)

### Deployment
- Vercel: `vercel` command ready to deploy gallery
- Full app: Use `docker compose -f docker-compose.gpu.yml up -d`

## Key Files Changed

```
src/ai_artist/curation/curator.py    # Added is_black_or_blank()
src/ai_artist/core/generator.py      # Filters invalid images
src/ai_artist/core/face_restore.py   # NEW - GFPGAN face restoration
src/ai_artist/gallery/manager.py     # Added cleanup_invalid_images()
src/ai_artist/web/app.py             # Concurrency, cleanup endpoint
config/config.yaml                   # float32, enhanced negative prompt
config/wildcards/quality.txt         # NEW
config/wildcards/face_quality.txt    # NEW
scripts/cleanup_gallery.py           # NEW
Dockerfile                           # Improved
Dockerfile.gpu                       # NEW
docker-compose.gpu.yml               # NEW
vercel.json                          # NEW
api/index.py                         # NEW
```

## Commands

```bash
# Clean gallery of black images
python scripts/cleanup_gallery.py

# Start web server
python -m uvicorn ai_artist.web.app:app --reload

# Deploy to Vercel
vercel

# Deploy with Docker + GPU
docker compose -f docker-compose.gpu.yml up -d

# Install face restoration
pip install gfpgan
```

## Next Steps

1. Wire face restoration into main.py pipeline
2. Test generation with new negative prompts
3. Consider switching to DreamShaper for better faces
4. Deploy gallery to Vercel or Railway
