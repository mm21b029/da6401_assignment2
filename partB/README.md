# iNaturalist Classification with Fine-Tuned ResNet50

## Prerequisites

- Python 3.7+
- PyTorch 1.12+
- PyTorch Lightning 2.0+
- Torchvision 0.13+
- Weights & Biases (`wandb`)

Install requirements:
```bash
pip install torch torchvision pytorch-lightning wandb
```

Place dataset in parent directory: ./inaturalist_12K/

## Training

```bash
python pretrained.py
```