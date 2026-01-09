# Training Data Acquisition and Preparation Guide

Complete guide for sourcing, preparing, and using training data for LoRA fine-tuning while maintaining legal compliance.

---

## Table of Contents

- [Legal Requirements](#legal-requirements)
- [Data Sources](#data-sources)
- [Dataset Preparation](#dataset-preparation)
- [Captioning Guidelines](#captioning-guidelines)
- [Quality Standards](#quality-standards)
- [Training Dataset Structure](#training-dataset-structure)
- [Best Practices](#best-practices)

---

## Legal Requirements

**⚠️ CRITICAL: Read LEGAL.md before collecting any training data!**

### Key Legal Principles

1. **Public Domain Only**
   - Use images in public domain or with explicit CC0 license
   - Verify license status for every image
   - Document all sources

2. **No Copyright Infringement**
   - Never train on copyrighted artwork without permission
   - Avoid recognizable artist styles
   - No celebrity faces or trademarked content

3. **Attribution Requirements**
   - Track source of every training image
   - Maintain metadata with license info
   - Be prepared to remove images on request

4. **EU AI Act Compliance (2026)**
   - Publish training data summary
   - Disclose data sources
   - Label all generated content as AI-created

---

## Data Sources

### Recommended Sources (Public Domain)

#### 1. **Unsplash API (Free for Training)**

**License**: Unsplash License (similar to CC0)
**Usage**: Allowed for ML training

**Access:**
```bash
# Already configured if you have API key
python scripts/download_unsplash.py --query "landscape" --count 100
```

**Guidelines:**
- Always attribute photographers
- Track download_location URLs
- Respect rate limits (50/hour demo, 5000/hour production)

---

#### 2. **Pexels API**

**License**: Pexels License (free for commercial use)
**Usage**: Allowed for training

**Access:**
```python
from pexels_api import API
api = API(api_key='YOUR_KEY')
api.search('nature', results_per_page=100)
```

---

#### 3. **WikiArt (Public Domain Only)**

**URL**: https://www.wikiart.org/
**License**: Filter for "Public Domain" only
**Usage**: Historical artwork (pre-1924)

**Warning**: Most art on WikiArt is still copyrighted. Only use:
- Artists who died before 1954 (70 years ago)
- Works explicitly marked "Public Domain"

---

#### 4. **Met Museum API**

**URL**: https://metmuseum.github.io/
**License**: CC0 for images marked "Public Domain"
**Usage**: Historical art collection

**Access:**
```bash
pip install met-museum-api
```

---

#### 5. **LAION Datasets**

**URL**: https://laion.ai/
**License**: Various (check metadata)
**Usage**: Large-scale datasets with license metadata

**Warning**: Filter for CC0/Public Domain only
```python
# Filter LAION for usable images
df = df[df['LICENSE'].isin(['CC0', 'Public Domain'])]
```

---

### Sources to AVOID

❌ **Google Images** - Mostly copyrighted
❌ **DeviantArt** - Copyrighted by artists
❌ **ArtStation** - Copyrighted professional work
❌ **Pinterest** - Unknown licensing
❌ **Instagram** - User-generated, copyrighted
❌ **Any artist's portfolio** - Unless explicit permission

---

## Dataset Preparation

### Recommended Dataset Size

**For LoRA Training:**
- **Minimum**: 15-20 images (proof of concept)
- **Recommended**: 50-100 images (good quality)
- **Optimal**: 100-200 images (best results)
- **Maximum**: 500+ images (risk of overfitting)

### Image Requirements

**Resolution:**
- SDXL: 1024x1024 pixels
- SD 1.5: 512x512 pixels

**Format:**
- PNG or JPEG
- RGB color space
- No alpha channel artifacts

**Content:**
- Clear, high-quality images
- Diverse compositions
- Consistent subject matter
- Good lighting and focus

---

### Preparation Script

**Step 1: Collect Raw Images**
```bash
mkdir -p raw_images
# Place your legally sourced images here
```

**Step 2: Run Preparation Script**
```bash
python scripts/prepare_training_data.py \
    --input raw_images/ \
    --output datasets/training/ \
    --resolution 1024
```

**What the script does:**
1. Resize images to 1024x1024 (square)
2. Convert to PNG
3. Create placeholder caption files
4. Generate metadata file

**Step 3: Edit Captions**
```bash
# Manually edit all .txt files
# See "Captioning Guidelines" below
```

---

### Regularization Dataset

**Purpose**: Prevent overfitting by providing examples of general style

**Creation:**
```bash
# Download 50-100 general images similar to your style
python scripts/download_unsplash.py \
    --query "artistic photography" \
    --count 100 \
    --output datasets/regularization/
```

**Captioning**: Use generic captions like "a photograph"

---

## Captioning Guidelines

### Why Captions Matter

Captions teach the model:
- What subjects to recognize
- What style keywords to associate
- How to interpret prompts

### Caption Format

**Pattern**: `[main subject], [details], [style keywords]`

**Examples:**

**Good Captions:**
```
a mountain landscape with snow-covered peaks, golden hour lighting, dramatic clouds
a portrait of a person in natural light, shallow depth of field, warm tones
a cityscape at night with illuminated buildings, long exposure, vibrant colors
```

**Bad Captions:**
```
landscape  # Too vague
beautiful mountain sunset  # Subjective adjectives
photo_1234.jpg  # Filename, not description
```

### Caption Length

- **Minimum**: 5-10 words
- **Optimal**: 10-20 words
- **Maximum**: 75 tokens (~50-60 words)

### Caption Consistency

**Use consistent terminology:**
- Same subject: "a portrait" not "a person" / "a face"
- Same style: "painterly style" across all images
- Same technical terms: "bokeh" not "blurry background"

### Activation Trigger (Optional)

Some trainers use a unique activation phrase:

```
[YOURSTYLE] a mountain landscape, dramatic lighting
[YOURSTYLE] a portrait, soft focus
```

**Pros**: Easy to trigger style
**Cons**: Less natural language, may overfit

---

## Quality Standards

### Image Quality Checklist

For each training image, verify:

- [ ] **Legal**: Confirmed public domain or CC0
- [ ] **Resolution**: At least 1024x1024
- [ ] **Focus**: Sharp, not blurry
- [ ] **Lighting**: Well-exposed, not over/underexposed
- [ ] **Composition**: Interesting, not cluttered
- [ ] **Subject**: Clearly identifiable
- [ ] **Diversity**: Adds variety to dataset

### Quality Filtering

**Remove images with:**
- Watermarks or text overlays
- Heavy compression artifacts
- Extreme crops or distortion
- Inappropriate content
- Duplicate or near-duplicate images

**Tool for Duplicate Detection:**
```python
import imagehash
from PIL import Image

def find_duplicates(image_paths, threshold=5):
    """Find perceptually similar images."""
    hashes = {}
    duplicates = []

    for path in image_paths:
        hash = imagehash.phash(Image.open(path))
        for existing_hash, existing_path in hashes.items():
            if hash - existing_hash < threshold:
                duplicates.append((path, existing_path))
        hashes[hash] = path

    return duplicates
```

---

## Training Dataset Structure

### Directory Layout

```
datasets/
├── training/
│   ├── 3_concept/              # Repeat 3 times per image
│   │   ├── image_001.png
│   │   ├── image_001.txt       # Caption
│   │   ├── image_002.png
│   │   ├── image_002.txt
│   │   └── ...
│   └── metadata.json           # Dataset metadata
│
├── regularization/
│   ├── 1_general/              # Repeat 1 time per image
│   │   ├── reg_001.png
│   │   ├── reg_001.txt
│   │   └── ...
│
└── validation/                 # Hold-out set for evaluation
    ├── val_001.png
    ├── val_001.txt
    └── ...
```

### Metadata File

**`metadata.json`:**
```json
{
    "dataset_name": "landscape_style_v1",
    "created_at": "2026-01-08",
    "total_images": 75,
    "resolution": "1024x1024",
    "sources": [
        {
            "source": "unsplash",
            "count": 50,
            "license": "Unsplash License",
            "query": "landscape mountains"
        },
        {
            "source": "met_museum",
            "count": 25,
            "license": "CC0",
            "collection": "American Landscapes 1800-1900"
        }
    ],
    "compliance": {
        "verified_public_domain": true,
        "attribution_file": "datasets/training/ATTRIBUTIONS.md",
        "removal_policy": "Contact project maintainer"
    }
}
```

### Attribution File

**`ATTRIBUTIONS.md`:**
```markdown
# Training Dataset Attributions

## Unsplash Images
- image_001.png: Photo by John Doe (https://unsplash.com/@johndoe)
- image_002.png: Photo by Jane Smith (https://unsplash.com/@janesmith)

## Met Museum Images
- image_051.png: "Mountain Landscape" by Artist Name (1850), Public Domain
- image_052.png: "Valley View" by Artist Name (1875), Public Domain

## License Summary
- Unsplash images: Unsplash License
- Met Museum images: CC0 (Public Domain)
```

---

## Best Practices

### Dataset Curation

1. **Start Small**
   - Begin with 20-30 high-quality images
   - Evaluate results before scaling up
   - Iterate on caption quality

2. **Diversity is Key**
   - Multiple angles of subjects
   - Various lighting conditions
   - Different compositions
   - Range of colors and moods

3. **Consistency Matters**
   - Similar subject matter
   - Consistent quality level
   - Unified style or aesthetic
   - Matching resolution

### Common Mistakes to Avoid

❌ **Too Few Images**: <15 images leads to overfitting
❌ **Too Many Images**: >500 without careful curation wastes compute
❌ **Inconsistent Captions**: Confuses model training
❌ **Low Quality Images**: Teaches model to produce poor output
❌ **Copyright Violations**: Legal risk, ethical issues
❌ **No Regularization**: Overfits to training set
❌ **Duplicate Images**: Wastes training steps

### Validation Set

**Purpose**: Evaluate model during training

**Size**: 10-20% of training set
**Selection**: Hold out diverse examples
**Usage**: Generate samples every 100 steps

**Example Prompts for Validation:**
```
a mountain landscape at sunset
a portrait in natural light
an urban cityscape at night
an abstract composition with vibrant colors
```

---

## Legal Documentation Template

### Training Data Report (EU AI Act)

**Required for public deployment:**

```markdown
# Training Data Report - [Model Name]

## Dataset Summary
- **Total Images**: 75
- **Date Range**: January 2026
- **Geographic Diversity**: International
- **Content Categories**: Landscapes, Nature, Mountains

## Data Sources
1. **Unsplash** (50 images)
   - License: Unsplash License (allows ML training)
   - URL: https://unsplash.com/
   - Verification: API downloads with attribution tracked

2. **Metropolitan Museum of Art** (25 images)
   - License: CC0 (Public Domain)
   - URL: https://www.metmuseum.org/
   - Verification: Filtered for "Public Domain" designation

## Copyright Compliance
- ✅ All images verified as public domain or explicitly licensed for ML
- ✅ No copyrighted artist styles mimicked
- ✅ No trademarked content included
- ✅ Attribution provided for all sources

## Bias Mitigation
- Diverse geographic representation
- Multiple perspectives and compositions
- Balanced color palettes and lighting conditions

## Removal Policy
Contact: [your email]
Process: Images removed within 7 days of request

## Audit Trail
- Dataset hash: [SHA256 of dataset]
- Training completed: 2026-01-08
- Model version: v1.0
```

---

## Dataset Quality Checklist

Before training, verify:

### Legal Compliance
- [ ] All images have verified public domain or CC0 license
- [ ] Attribution file created and complete
- [ ] No copyrighted content included
- [ ] Training data report prepared

### Technical Quality
- [ ] All images are 1024x1024 (or 512x512 for SD 1.5)
- [ ] No compression artifacts visible
- [ ] All images sharp and well-exposed
- [ ] No watermarks or text overlays

### Dataset Composition
- [ ] 50-100 training images
- [ ] 50-100 regularization images
- [ ] 10-20 validation images
- [ ] Good diversity in composition
- [ ] Consistent subject matter

### Captions
- [ ] All images have caption files (.txt)
- [ ] Captions are descriptive (10-20 words)
- [ ] Consistent terminology used
- [ ] No generic/placeholder captions

### Documentation
- [ ] metadata.json created
- [ ] ATTRIBUTIONS.md created
- [ ] Sources documented
- [ ] Training data report prepared

---

## Tools and Scripts

### Download Script Template

```python
"""Download and prepare images from Unsplash."""

import httpx
import asyncio
from pathlib import Path
from PIL import Image
import io

async def download_image(url: str, save_path: Path):
    """Download image from URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        image = image.convert("RGB")
        image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        image.save(save_path)

# Usage
asyncio.run(download_image(photo_url, Path("image.png")))
```

### Caption Generation Helper

```python
"""Generate caption templates for manual editing."""

import clip
import torch
from PIL import Image

def suggest_caption(image_path: Path) -> str:
    """Use CLIP to suggest initial caption."""
    model, preprocess = clip.load("ViT-B/32")

    image = preprocess(Image.open(image_path)).unsqueeze(0)

    # Define candidate descriptions
    candidates = [
        "a landscape",
        "a portrait",
        "an urban scene",
        "a nature photograph",
        "an abstract composition",
    ]

    text = clip.tokenize(candidates)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    best_match = candidates[similarity.argmax()]

    return f"{best_match}, [add details here]"
```

---

## Resources

### Legal Resources
- [Creative Commons Search](https://search.creativecommons.org/)
- [Public Domain Sherpa](https://www.publicdomainsherpa.com/)
- [Copyright Term and the Public Domain](https://copyright.cornell.edu/publicdomain)

### Image Sources
- [Unsplash API Docs](https://unsplash.com/documentation)
- [Pexels API Docs](https://www.pexels.com/api/documentation/)
- [Met Museum API](https://metmuseum.github.io/)
- [Library of Congress](https://www.loc.gov/pictures/)

### Tools
- [ImageMagick](https://imagemagick.org/) - Batch processing
- [imagehash](https://github.com/JohannesBuchner/imagehash) - Duplicate detection
- [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator) - Caption generation

---

**Document Version:** 1.0
**Last Updated:** 2026-01-08
**Critical Warning:** Always consult LEGAL.md before using any training data
