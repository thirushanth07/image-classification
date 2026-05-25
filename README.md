# image-classification

Deep Learning image classification project using **TensorFlow/Keras** and a **MobileNetV2 pretrained backbone** on **CIFAR-10**.

## Features
- Transfer learning with MobileNetV2 (`imagenet` weights)
- Data augmentation (flip/rotation/zoom)
- Training + validation curves output
- Evaluation metrics (test accuracy/loss + per-class precision/recall/F1)
- Saved model inference on CIFAR-10 samples or custom images

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow matplotlib numpy
```

## Train
```bash
python train_mobilenetv2_cifar10.py --mode train --epochs 5 --fine-tune-epochs 2
```

Training artifacts:
- `outputs/metrics.json`
- `outputs/training_curves.png`
- `saved_model/mobilenetv2_cifar10`

## Quick smoke run (faster)
```bash
python train_mobilenetv2_cifar10.py --mode train --epochs 1 --fine-tune-epochs 0 --max-train-samples 1024
```

## Inference
From saved model with CIFAR-10 test image:
```bash
python train_mobilenetv2_cifar10.py --mode infer --model-path saved_model/mobilenetv2_cifar10 --sample-index 0
```

From saved model with a custom image:
```bash
python train_mobilenetv2_cifar10.py --mode infer --model-path saved_model/mobilenetv2_cifar10 --infer-image ./images/sample.png
```
