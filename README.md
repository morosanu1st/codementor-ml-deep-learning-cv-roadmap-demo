# ML / Deep Learning / Computer Vision mentoring demo

A small, runnable demo from **Mihai Chindriș** (Codementor mentor) for a long-term mentee who
already knows the AI basics and wants to go *deep*: machine learning, deep learning, computer
vision and model architecture.

The idea is simple: before we lean on PyTorch, you should be able to write the core pieces
yourself in NumPy. Then the framework stops being magic.

## What the code shows

| File | What you learn |
|------|----------------|
| `src/mini_nn.py` | A 2-layer neural net in pure NumPy: forward pass, softmax + cross-entropy, **backprop by hand**, gradient descent. Trained on a synthetic "two moons" dataset, which is not linearly separable, and gets about 97% accuracy. |
| `src/conv_from_scratch.py` | 2D convolution (with stride and padding), ReLU and max-pooling from scratch. Applies Sobel filters to a toy image and prints the **feature maps**, so you can see what a CNN's first layer detects. |
| `notes/architecture_building_blocks.md` | Short notes on linear layers, convs, normalisation, **residual blocks**, **attention / Transformers**, ViT and U-Net, plus an architecture-design checklist. |
| `tests/` | pytest tests, including a **numerical gradient check** that proves the hand-written backprop is correct. |

## How to run

```bash
pip install numpy pytest

python -m src.mini_nn            # trains the network, prints loss / accuracy
python -m src.conv_from_scratch  # prints input image, feature maps, pooled map
pytest                           # runs all tests
```

Python 3.9+.

## Long-term mentoring roadmap

You can move faster or slower depending on your background. Every phase ends with a small project
you build yourself.

| # | Phase | Key topics | Milestone project |
|---|-------|-----------|-------------------|
| 1 | **Math refresh** | Linear algebra (matmul, shapes, eigen/SVD), derivatives and the chain rule, probability, MLE | Vectorise a loop-heavy NumPy script |
| 2 | **Classic ML** | Linear/logistic regression, regularisation, trees and gradient boosting, bias/variance, cross-validation, metrics | End-to-end tabular model with a proper validation setup |
| 3 | **Neural nets from scratch** | This repo: backprop, initialisation, optimisers (SGD, momentum, Adam), gradient checking | Add a hidden layer and Adam to `mini_nn.py` |
| 4 | **PyTorch fundamentals** | Tensors, autograd, `nn.Module`, `DataLoader`, training loops, GPU, debugging (overfitting one batch) | Re-implement `mini_nn` in PyTorch and match the results |
| 5 | **CNNs & computer vision** | Conv/pool, receptive field, BatchNorm, augmentation, ResNet, transfer learning | Fine-tune a ResNet on a custom image dataset |
| 6 | **Detection & segmentation** | Anchors vs anchor-free, IoU/NMS, YOLO / Faster R-CNN, U-Net, Mask R-CNN, mAP | Train a small detector or segmenter on your own data |
| 7 | **Transformers & ViT** | Attention from scratch, positional encodings, ViT, Swin, CLIP-style pre-training | Write a mini-ViT and compare it with a CNN |
| 8 | **Model architecture design** | Residual/attention trade-offs, scaling laws, efficiency (depthwise conv, MobileNet), ablation studies | Design, ablate and write up your own architecture |
| 9 | **Training at scale & experiment hygiene** | LR schedules, mixed precision, experiment tracking, reproducibility, reading papers | Reproduce one figure from a paper |
| 10 | **Deployment** | ONNX / TorchScript, quantisation, FastAPI serving, edge inference, monitoring | Ship a CV model behind an API |

## How sessions work

- **Cadence:** 1–2 live sessions a week (60–90 min), plus async code review between sessions.
- **Format:** a short concept explanation, then live coding together, then you build the next part on your own.
- **Homework:** one small, testable task per session, pushed to your own GitHub repo, which I review.
- **Paper track:** from phase 5 on, we read one key paper every 1–2 weeks (ResNet, U-Net, YOLO, *Attention Is All You Need*, ViT…) and implement its core idea.
- **Checkpoints:** every ~4 weeks we review progress and adjust the roadmap to your goals (research, industry CV, or ML engineering).

---
*Codementor demo by Mihai Chindriș*
