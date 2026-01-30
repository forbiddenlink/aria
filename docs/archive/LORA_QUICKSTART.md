# LoRA Training & Management Quick Start

This guide shows you how to train and use multiple specialized LoRA styles for different use cases.

## 🎨 Three Specialized Styles

### 1. **Abstract Art** 
- Colorful, artistic compositions
- Perfect for: Creative projects, artistic designs, vibrant artwork

### 2. **Landscape Photography**
- Dramatic natural scenery
- Perfect for: Nature backgrounds, travel imagery, outdoor scenes

### 3. **Web Hero Images** ⭐ (Perfect for Web Development!)
- Professional, clean, modern compositions (16:9 format)
- Perfect for: Website headers, landing pages, hero sections, marketing materials

## 🚀 Complete Workflow

### Step 1: Download Training Data

Download datasets for all three styles:
```bash
python scripts/download_specialized_datasets.py all --num-images 30
```

Or download individually:
```bash
# Abstract art (colorful, artistic)
python scripts/download_specialized_datasets.py abstract --num-images 30

# Landscape photography (dramatic scenery)
python scripts/download_specialized_datasets.py landscape --num-images 30

# Web hero images (professional, 16:9 format)
python scripts/download_specialized_datasets.py webhero --num-images 30
```

### Step 2: Train LoRAs

Train all three styles (takes 4-6 hours total):
```bash
python scripts/train_all_loras.py all
```

Or train individually (~1.5-2 hours each):
```bash
python scripts/train_all_loras.py abstract
python scripts/train_all_loras.py landscape
python scripts/train_all_loras.py webhero
```

### Step 3: Switch Between LoRAs

**List available LoRAs:**
```bash
python scripts/manage_loras.py list
```

**Check current status:**
```bash
python scripts/manage_loras.py status
```

**Activate a LoRA:**
```bash
# For abstract art
python scripts/manage_loras.py set abstract_style --scale 0.8

# For landscape photography
python scripts/manage_loras.py set landscape_style --scale 0.7

# For web hero images (your web dev projects!)
python scripts/manage_loras.py set webhero_style --scale 0.8

# Disable LoRA (use base model only)
python scripts/manage_loras.py set none
```

### Step 4: Generate Images

After activating a LoRA, generate images as normal:
```bash
ai-artist
```

Or use the web interface:
```bash
ai-artist web
```

## 🎯 Use Cases

### For Web Development (webhero_style)

Perfect for creating professional website hero images:

```bash
# Activate web hero LoRA
python scripts/manage_loras.py set webhero_style --scale 0.8

# Generate hero images
ai-artist
```

**Prompt ideas for web heroes:**
- "modern tech office with computers"
- "professional business team working together"
- "sleek product on minimalist background"
- "laptop on wooden desk with coffee cup"
- "futuristic digital interface"
- "clean workspace with natural lighting"

### For Artistic Projects (abstract_style)

```bash
python scripts/manage_loras.py set abstract_style --scale 0.8
```

**Prompt ideas:**
- "flowing colors and shapes"
- "geometric patterns"
- "vibrant energy"
- "dreamlike composition"

### For Nature/Travel (landscape_style)

```bash
python scripts/manage_loras.py set landscape_style --scale 0.7
```

**Prompt ideas:**
- "mountain peak at golden hour"
- "misty forest in morning light"
- "dramatic coastal cliffs"
- "desert dunes at sunset"

### For General Purpose (no LoRA)

```bash
python scripts/manage_loras.py set none
```

Maximum flexibility - can generate anything!

## ⚙️ Advanced: LoRA Scale Control

The `--scale` parameter controls how strongly the LoRA affects generation:

- **0.0**: No effect (same as disabled)
- **0.3**: Subtle influence - mostly base model
- **0.5**: Balanced mix
- **0.7**: Strong style influence
- **0.8**: Very strong style (recommended)
- **1.0**: Maximum strength

Example:
```bash
# Subtle web hero influence
python scripts/manage_loras.py set webhero_style --scale 0.3

# Strong abstract art style
python scripts/manage_loras.py set abstract_style --scale 0.9
```

## 📊 Training Parameters

Each style uses optimized training parameters:

| Style | Rank | Steps | Learning Rate | Resolution | Training Time |
|-------|------|-------|---------------|------------|---------------|
| Abstract | 16 | 3000 | 5e-5 | 512x512 | ~2 hours |
| Landscape | 8 | 3000 | 1e-4 | 512x512 | ~2 hours |
| Web Hero | 8 | 2500 | 1e-4 | 768x768 | ~1.5 hours |

## 🔄 Workflow Example: Building a Website

1. **Download web hero dataset:**
   ```bash
   python scripts/download_specialized_datasets.py webhero --num-images 30
   ```

2. **Train web hero LoRA:**
   ```bash
   python scripts/train_all_loras.py webhero
   ```

3. **Activate for use:**
   ```bash
   python scripts/manage_loras.py set webhero_style --scale 0.8
   ```

4. **Generate hero images:**
   ```bash
   ai-artist
   ```
   Prompts: "modern tech workspace", "professional team meeting", etc.

5. **Switch to abstract for decorative elements:**
   ```bash
   python scripts/manage_loras.py set abstract_style
   ai-artist
   ```
   Prompts: "colorful geometric background", "flowing wave pattern"

6. **Back to base model for specific needs:**
   ```bash
   python scripts/manage_loras.py set none
   ai-artist
   ```
   Prompts: Any custom requirement

## 🎓 Tips

- **Web developers**: The webhero LoRA creates professional 16:9 compositions perfect for headers
- **Start with scale 0.8**: This gives strong style influence while maintaining flexibility
- **Test without LoRA first**: Compare base model output vs LoRA-enhanced
- **Mix and match**: Use different LoRAs for different sections of your project
- **Save your favorites**: Keep track of which scale works best for each use case

## 📁 File Locations

- **Datasets**: `datasets/training_abstract/`, `datasets/training_landscape/`, `datasets/training_webhero/`
- **Trained LoRAs**: `models/lora/abstract_style/`, `models/lora/landscape_style/`, `models/lora/webhero_style/`
- **Config**: `config/config.yaml`
- **Generated images**: `gallery/`

## 🆘 Troubleshooting

**Training fails:**
- Check dataset has 20+ images
- Ensure enough disk space (~5GB per LoRA)

**LoRA not working:**
- Verify path with: `python scripts/manage_loras.py status`
- Check LoRA directory exists and has model files
- Try increasing scale to 0.9

**Wrong style output:**
- Check current LoRA: `python scripts/manage_loras.py status`
- Deactivate and try again: `python scripts/manage_loras.py set none`
