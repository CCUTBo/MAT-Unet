# # MAT-Unet：A Hybrid Transformer Network with Residual-like Connections for Medical Image Segmentation

## 🧠 Overview

**MAT-Unet** is a novel medical image segmentation framework that incorporates a

- `Mix-Attention Transformer (MAT)` module for long-range context modeling,
- `Local Gaussian Self-Attention (LGSA)` for efficient spatial attention,
- `Multi-Head Self-Attention (MSA)` for semantic diversity,
- and `ResPath` skip connections for enhanced semantic bridging.

The model achieves **state-of-the-art results** on multi-modal datasets (CT, MRI, RGB), including:

- 🧠 ACDC (MRI)
- 🩻 Synapse (CT)
- 🧴 ISIC2017 (skin)
- 🧬 Kvasir-SEG (endoscopy)

## 📦 Installation

```shell
git clone https://github.com/yourname/MAT-Unet.git
cd MAT-Unet
pip install -r requirements.txt
```

Requires: `Python ≥ 3.8`, `PyTorch ≥ 1.10`, `torchvision`, etc.

## 🗂️ Dataset Preparation

```
data/
├── Synapse/
│   ├── lists_Synapse/
	│   ├── test.txt/
	│   └── train.txt/
│   ├── test/
│   └── train/
├── ACDC/
│   ├── lists_ACDC/
	│   ├── test.txt/
	│   ├── train.txt/
	│   └── valid.txt/
│   ├── test/
│   ├── train/
│   └── valid/
├── ISIC2017/
│   ├── test/
	│   ├── images/
	│   └── masks/
│   ├── train/
	│   ├── images/
	│   └── masks/
│   └── valid/
	│   ├── images/
	│   └── masks/
├── Kvasir/
│   ├── images/
│   ├── masks/
│   ├── train.txt/
│   └── val.txt/

```

## 🚀 Training

```shell
python train.py --dataset Synapse --img_size 224 --batch_size 12 --lr 1e-4
```

Optional arguments:

`--dataset [ACDC | ISIC2017 | Kvasir]`

## 🔍 Evaluation

```shell
python test.py --dataset Synapse --checkpoint ./checkpoints/best.pth
```

## 📊 Results
