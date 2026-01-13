# Model Weights Storage

This repository stores trained model weights for HPS-Seg and H-CLIP projects.

## Directory Structure

```
models/
├── hps_seg/
│   ├── enhanced_catseg/     # HPS-Seg Enhanced Model (with HPA + AFR)
│   └── h_clip_vitb/         # H-CLIP Baseline Model
```

## Model Details

### HPS-Seg Enhanced Model (`enhanced_catseg/`)

- **Model**: HPS-Seg with Hyperspherical Prototype Alignment (HPA) and Adaptive Feature Rectification (AFR)
- **Backbone**: ViT-B/16
- **Checkpoint**: `model_0029999.pth` (latest, 30K iterations)
- **Training**: Trained on COCO-Stuff dataset
- **Files**:
  - `model_0019999.pth` - 20K iterations checkpoint
  - `model_0024999.pth` - 25K iterations checkpoint
  - `model_0029999.pth` - 30K iterations checkpoint (latest)

### H-CLIP Baseline Model (`h_clip_vitb/`)

- **Model**: H-CLIP Baseline (OFT)
- **Backbone**: ViT-B/16
- **Checkpoint**: `model_0009999.pth` (latest, 10K iterations)
- **Training**: Trained on COCO-Stuff dataset
- **Files**:
  - `model_0004999.pth` - 5K iterations checkpoint
  - `model_0009999.pth` - 10K iterations checkpoint (latest)

## Usage

### Download Models

You can download specific model checkpoints using Git LFS or direct download from GitHub releases.

### Load Model in Code

```python
import torch
from cat_seg import build_model

# Load HPS-Seg Enhanced model
checkpoint = torch.load('models/hps_seg/enhanced_catseg/model_0029999.pth')
model = build_model(checkpoint['cfg'])
model.load_state_dict(checkpoint['model'])

# Load H-CLIP Baseline model
checkpoint = torch.load('models/hps_seg/h_clip_vitb/model_0009999.pth')
model = build_model(checkpoint['cfg'])
model.load_state_dict(checkpoint['model'])
```

## File Sizes

- Each checkpoint file: ~620MB
- Total repository size: ~5GB

## Notes

- Models are stored using Git LFS for efficient version control
- Latest checkpoints are recommended for inference
- Intermediate checkpoints are provided for analysis and comparison

## Related Repositories

- [HPS-Seg](https://github.com/ZhaoYi-10-13/HPS-Seg) - Main repository with training code
- [H-CLIP](https://github.com/ZhaoYi-10-13/H-CLIP) - Baseline implementation
