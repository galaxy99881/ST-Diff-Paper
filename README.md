# ST-Diff: Mask-Conditioned Breast Ultrasound Image Generation Framework

**ST-Diff** is a mask-conditioned breast ultrasound image generation framework for medical education and segmentation training. This repository contains the implementation code for the research paper.

## Overview

ST-Diff addresses the dual challenges in breast ultrasound imaging:
1. **Medical Education**: Limited case diversity for systematic training of complex lesions
2. **AI Development**: Scarcity of high-quality "image-mask-text" paired data with expensive pixel-level annotations

## Key Features

- **Mask-Conditioned Generation**: Explicit foreground-background decoupling using masks as hard constraints
- **Dual-Branch Architecture**: 
  - ControlNet module for foreground structural reconstruction
  - CLIP-LoRA adapter for foreground semantic adaptation
- **BI-RADS Semantic Control**: Precise injection of clinical features into lesion regions
- **Two-Stage Training**: Unconditional pretraining + multi-condition fine-tuning

## Architecture

```
ST-Diff
├── Latent Diffusion Model (LDM) [Base]
├── ControlNet Branch [Structure]
│   └── Zero Convolution Injection
└── CLIP-LoRA Branch [Semantics]
    └── BI-RADS Feature Injection
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ST-Diff.git
cd ST-Diff

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Prepare your dataset in the following structure:
```
data/
├── unlabeled/          # Unlabeled images for pretraining
│   └── images/
└── labeled/            # Labeled triplets (image-mask-text)
    ├── images/
    ├── masks/
    └── texts/
```

### 2. Training

**Stage 1: Unconditional Pretraining**
```bash
python train.py --stage pretrain --config configs/pretrain_config.yaml
```

**Stage 2: Multi-Condition Fine-tuning**
```bash
python train.py --stage finetune --config configs/finetune_config.yaml --pretrained_path checkpoints/pretrain/latest.ckpt
```

### 3. Inference

Generate images from mask and text:
```bash
python inference.py \
    --mask_path path/to/mask.png \
    --text "BI-RADS 4: irregular shape, indistinct margin, hypoechoic" \
    --output_path output/generated_image.png \
    --checkpoint checkpoints/finetune/best.ckpt
```

## Evaluation

### Quantitative Metrics
```bash
python evaluate.py \
    --test_data data/test \
    --checkpoint checkpoints/finetune/best.ckpt \
    --metrics fid ssim lpips mask_dice birads_acc
```

### Turing Test
```bash
python turing_test.py \
    --real_images data/test/images \
    --generated_images outputs/generated \
    --expert_raters expert_list.csv
```

## Results

- **FID**: 18.2 ± 0.3
- **Mask Dice**: 0.82 ± 0.02
- **BI-RADS Accuracy**: 0.83 ± 0.02
- **Turing Test Misclassification Rate**: 91.6% ± 2.1%

## Citation

If you use this code in your research, please cite:

```bibtex
@article{st-diff2025,
  title={ST-Diff: A Mask-Conditioned Breast Ultrasound Image Generation Framework for Medical Education and Segmentation Training},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Latent Diffusion Models (Stable Diffusion)
- ControlNet for structural control
- CLIP-LoRA for semantic adaptation
- Public datasets: BUSI, BUS-BRA, BUS-CoT

