# Training Data Sourcing Guide

## ⚠️ CRITICAL: Legal Compliance

**MUST READ**: Review [LEGAL.md](LEGAL.md) before sourcing any training data.

All training images must be:

- Public domain OR
- Licensed under CC0, CC BY, CC BY-SA OR
- Personally created by you OR
- Licensed with explicit permission for AI training

## Recommended Sources

### 1. Public Domain Collections

#### Wikimedia Commons

- URL: <https://commons.wikimedia.org>
- License Filter: Public domain / CC0
- Categories: Art, Nature, Architecture
- Quality: High-resolution scans available
- **Process**: Search → Filter by license → Download

#### Metropolitan Museum of Art

- URL: <https://www.metmuseum.org/art/collection>
- License: CC0 for Open Access artworks
- Quantity: 400,000+ images
- Quality: Professional photography
- **Process**: Use "Open Access" filter

#### Smithsonian Open Access

- URL: <https://www.si.edu/openaccess>
- License: CC0
- Quantity: 3 million+ images
- Quality: Museum-quality scans

#### Rijksmuseum

- URL: <https://www.rijksmuseum.nl/en/rijksstudio>
- License: Public domain
- Focus: Dutch Masters
- Quality: Ultra high-resolution

### 2. Creative Commons Platforms

#### Unsplash

- URL: <https://unsplash.com>
- License: Unsplash License (AI training allowed)
- Quantity: Millions of photos
- Quality: Professional photography
- **Note**: Check individual licenses

#### Pexels

- URL: <https://www.pexels.com>
- License: Pexels License (free for AI training)
- Quantity: Large collection
- Quality: High-quality stock photos

#### Pixabay

- URL: <https://pixabay.com>
- License: Pixabay License (free for any use)
- Quantity: 2.5 million+ images

### 3. Government Archives

#### NASA Image Library

- URL: <https://images.nasa.gov>
- License: Public domain (US Government)
- Focus: Space, science, astronomy
- Quality: Exceptional

#### Library of Congress

- URL: <https://www.loc.gov/collections>
- License: Many public domain items
- Quality: Historical significance

## Dataset Requirements

### Quantity

- **Minimum**: 20 images
- **Recommended**: 30-50 images
- **Maximum**: 100 images (risk of overfitting)

### Quality Standards

- **Resolution**: Minimum 512x512 pixels
- **Format**: JPEG or PNG
- **Content**: Clear, well-composed
- **Variety**: Different subjects/compositions
- **Consistency**: Similar artistic style

### Content Selection

#### For Style Training

Focus on images that share:

- Similar color palette
- Consistent artistic technique
- Common mood/atmosphere
- Recognizable aesthetic

Examples:

- All impressionist paintings
- All black & white photography
- All watercolor illustrations
- All cyberpunk cityscapes

#### What to Avoid

- Mixed styles in one dataset
- Low resolution or blurry images
- Watermarked images
- Copyrighted content without permission
- Images with visible people (privacy concerns)

## Preparation Checklist

- [ ] Source 20-50 images from legal sources
- [ ] Document source URL for each image
- [ ] Record license type for each image
- [ ] Create licenses.txt file with attributions
- [ ] Resize images to 512x512 if needed
- [ ] Check image quality (no artifacts)
- [ ] Verify style consistency
- [ ] Review LEGAL.md compliance

## Directory Structure

```
datasets/
├── training/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── regularization/  # Optional
│   ├── reg_001.jpg
│   └── ...
└── licenses.txt     # REQUIRED
```

## License Documentation Format

Create `datasets/licenses.txt`:

```
Image: image_001.jpg
Source: https://commons.wikimedia.org/wiki/File:Example.jpg
License: Public Domain
Artist: John Doe
Date: 1890

Image: image_002.jpg
Source: https://unsplash.com/photos/abc123
License: Unsplash License
Photographer: Jane Smith
Date: 2023

...
```

## Training Command

Once dataset is prepared:

```bash
python -m ai_artist.training.train_lora \
    --instance_data_dir datasets/training \
    --output_dir models/lora/my_style \
    --rank 4 \
    --max_train_steps 2000
```

## Validation

Before training:

1. Run license audit: Review all sources
2. Check image quality: All images load correctly
3. Verify consistency: Style is cohesive
4. Count images: 20-50 range
5. Document everything: licenses.txt complete

## Resources

- [Creative Commons License Chooser](https://creativecommons.org/choose/)
- [Public Domain Review](https://publicdomainreview.org/)
- [Open Culture Collections](https://www.openculture.com/)

## Questions?

Refer to [LEGAL.md](LEGAL.md) for detailed legal guidelines.
