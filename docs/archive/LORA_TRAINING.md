# LoRA Training Guide

## Overview

This guide walks you through training a custom LoRA (Low-Rank Adaptation) style for the AI Artist system. LoRA enables lightweight fine-tuning of Stable Diffusion to create unique artistic styles.

## Prerequisites

1. **Training data**: 20-50 images following guidelines in [TRAINING_DATA_SOURCING.md](TRAINING_DATA_SOURCING.md)
2. **Dependencies installed**: `pip install -e .` (includes peft, accelerate)
3. **Legal compliance**: Review [LEGAL.md](LEGAL.md) before sourcing data

## Quick Start

### 1. Prepare Your Dataset

```bash
# Place training images in datasets/training/
cp your_images/*.jpg datasets/training/

# Document sources in datasets/licenses.txt
# See TRAINING_DATA_SOURCING.md for template
```

### 2. Train Your LoRA

```bash
# Basic training (recommended for first time)
python -m ai_artist.training.train_lora \
    --instance_data_dir datasets/training \
    --output_dir models/lora/my_style \
    --rank 4 \
    --max_train_steps 2000

# Advanced training with custom parameters
python -m ai_artist.training.train_lora \
    --instance_data_dir datasets/training \
    --output_dir models/lora/my_style \
    --rank 8 \
    --learning_rate 1e-4 \
    --max_train_steps 5000 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --resolution 512
```

### 3. Use Your Trained Style

Edit `config/config.yaml`:

```yaml
model:
  base_model: "runwayml/stable-diffusion-v1-5"
  device: "mps"  # or "cuda" or "cpu"
  dtype: "float32"  # Use float32 on Apple Silicon
  lora_path: "models/lora/my_style"  # Path to your trained LoRA
  lora_scale: 0.8  # Strength (0.0-1.0)
```

Then generate:

```bash
ai-artist
```

## Training Parameters Explained

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--instance_data_dir` | Required | Directory with training images |
| `--output_dir` | `models/lora` | Where to save trained LoRA |
| `--rank` | 4 | LoRA rank (4-128, lower=faster) |
| `--max_train_steps` | 2000 | Training iterations |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | 1e-4 | Learning rate (1e-5 to 1e-3) |
| `--train_batch_size` | 1 | Batch size (GPU memory dependent) |
| `--gradient_accumulation_steps` | 4 | Effective batch size multiplier |
| `--resolution` | 512 | Image resolution for training |
| `--mixed_precision` | no | Use fp16/bf16 (GPU only) |

### Parameter Tuning Guide

**LoRA Rank**:
- `rank=4`: Fast training, subtle style (recommended start)
- `rank=8`: Balanced quality/speed
- `rank=16-32`: Strong style, slower training
- `rank=64+`: Very strong style, risk of overfitting

**Training Steps**:
- `2000`: Quick iteration (20-40 min on MPS)
- `3000-4000`: Standard quality
- `5000+`: Maximum quality, risk of overfitting

**Learning Rate**:
- `1e-5`: Conservative, slow learning
- `1e-4`: Default, balanced
- `5e-4`: Aggressive, faster convergence
- `1e-3`: Very aggressive, risk of instability

## Training Time Estimates

On Apple Silicon (M1/M2/M3):
- 2000 steps: ~20-40 minutes
- 5000 steps: ~50-100 minutes

On NVIDIA GPU (3090/4090):
- 2000 steps: ~10-20 minutes
- 5000 steps: ~25-50 minutes

## Monitoring Training

Watch for these logs:

```
training_started: model=..., rank=4, steps=2000
Steps: 10%|████      | 100/2000 [03:45<33:45, 1.07s/it, loss=0.234, lr=0.0001]
training_progress: step=100, loss=0.234
...
training_complete: output_dir=models/lora/my_style
```

**Good signs**:
- Loss decreasing over time
- Loss stabilizes around 0.05-0.15
- No NaN or infinite values

**Warning signs**:
- Loss increasing or oscillating wildly
- Loss goes to 0.00 (overfitting)
- NaN/Inf values (reduce learning rate)

## Testing Your LoRA

### Method 1: Update Config

```yaml
# config/config.yaml
model:
  lora_path: "models/lora/my_style"
  lora_scale: 0.8
```

Run: `ai-artist`

### Method 2: Manual Testing

```python
from ai_artist.core.generator import ImageGenerator
from pathlib import Path

generator = ImageGenerator(
    model_id="runwayml/stable-diffusion-v1-5",
    device="mps",
    dtype=torch.float32,
)
generator.load_model()
generator.load_lora(Path("models/lora/my_style"), lora_scale=0.8)

images = generator.generate(
    prompt="a serene mountain landscape",
    num_images=3,
)
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce batch size: `--train_batch_size 1`
2. Reduce resolution: `--resolution 512`
3. Reduce rank: `--rank 4`
4. Close other applications

### Issue: Loss Not Decreasing

**Solutions**:
1. Check training data quality
2. Increase learning rate: `--learning_rate 5e-4`
3. Train for more steps: `--max_train_steps 3000`
4. Verify dataset has consistent style

### Issue: Overfitting (Loss = 0.00)

**Solutions**:
1. Reduce training steps
2. Add more diverse training images
3. Lower learning rate
4. Reduce LoRA rank

### Issue: Style Not Strong Enough

**Solutions**:
1. Increase LoRA scale in config: `lora_scale: 1.0`
2. Increase training steps: `--max_train_steps 5000`
3. Increase LoRA rank: `--rank 8`
4. Ensure training data has consistent style

### Issue: Style Too Strong / Overpowering

**Solutions**:
1. Decrease LoRA scale: `lora_scale: 0.5`
2. Reduce training steps
3. Reduce LoRA rank: `--rank 4`
4. Add more variety to training data

## Best Practices

### Data Preparation

1. **Consistency**: All images should share artistic style
2. **Quality**: High resolution (min 512x512)
3. **Quantity**: 30-50 images ideal
4. **Diversity**: Varied subjects, same style
5. **Legal**: Document all sources

### Training Strategy

1. **Start small**: Begin with rank=4, 2000 steps
2. **Iterate**: Test output, adjust parameters
3. **Document**: Track what works
4. **Validate**: Test on diverse prompts
5. **Version**: Save different training runs

### Production Use

1. **Test thoroughly**: Generate 10+ images
2. **Compare**: Base model vs LoRA
3. **Adjust scale**: Find optimal strength
4. **Monitor**: Check for artifacts
5. **Backup**: Save successful LoRA weights

## Example Workflows

### Workflow 1: Impressionist Paintings

```bash
# 1. Source 30 impressionist paintings from Wikimedia Commons
# 2. Place in datasets/training/
# 3. Train
python -m ai_artist.training.train_lora \
    --instance_data_dir datasets/training \
    --output_dir models/lora/impressionist \
    --rank 8 \
    --max_train_steps 3000

# 4. Configure
# config.yaml: lora_path: "models/lora/impressionist"

# 5. Generate
ai-artist
```

### Workflow 2: Cyberpunk Style

```bash
# 1. Source 40 cyberpunk cityscape photos from Unsplash
# 2. Train with higher rank for strong style
python -m ai_artist.training.train_lora \
    --instance_data_dir datasets/training \
    --output_dir models/lora/cyberpunk \
    --rank 16 \
    --learning_rate 1e-4 \
    --max_train_steps 4000

# 3. Use with medium strength
# config.yaml: lora_path: "models/lora/cyberpunk", lora_scale: 0.7
```

## Advanced Topics

### Multiple LoRAs

Train multiple styles for different moods:

```bash
# Style 1: Serene landscapes
python -m ai_artist.training.train_lora \
    --instance_data_dir datasets/landscapes \
    --output_dir models/lora/serene_landscape

# Style 2: Vibrant abstracts
python -m ai_artist.training.train_lora \
    --instance_data_dir datasets/abstracts \
    --output_dir models/lora/vibrant_abstract
```

Switch by updating config.yaml.

### Regularization (Advanced)

Prevent overfitting with regularization images:

```bash
python -m ai_artist.training.train_lora \
    --instance_data_dir datasets/training \
    --regularization_data_dir datasets/regularization \
    --output_dir models/lora/my_style
```

Regularization images should be general photos in the same domain.

## Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Diffusers LoRA Training](https://huggingface.co/docs/diffusers/training/lora)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Training Data Guide](TRAINING_DATA_SOURCING.md)
- [Legal Guidelines](LEGAL.md)

## Next Steps

After successful LoRA training:

1. **Automate**: Set up scheduled generation (Phase 3)
2. **Expand**: Train multiple style variations
3. **Experiment**: Try different subjects with your style
4. **Share**: Document your artistic process
5. **Iterate**: Continuously refine your style

Happy training! 🎨
