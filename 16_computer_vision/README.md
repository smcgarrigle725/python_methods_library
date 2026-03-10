# 16 — Computer Vision

Applied computer vision with PyTorch: preprocessing and augmentation pipelines, classification with custom and pretrained architectures, object detection, semantic segmentation with U-Net, and rigorous evaluation. All examples use simulated ecological imagery — satellite-like land-cover patches and annotated field sensor images.

---

## Notebooks

### `image_preprocessing.ipynb`
`torchvision.transforms` pipeline: Resize, RandomHorizontalFlip, ColorJitter, Rotation, ToTensor, Normalize. ImageNet mean/std constants. Custom `Dataset` class with separate train/val transforms. Augmentation type visualisation (HFlip, VFlip, ColorJitter, Rotation, GaussianBlur). Normalisation effect on training stability via SGD comparison. Pitfalls: augmentation on val/test, label-changing transforms, normalisation mismatch, ToTensor before Normalize, `num_workers` on Windows.

### `image_classification.ipynb`
`ResBlock` (skip connection). `MiniResNet` with stem → residual layers → AdaptiveAvgPool → head. AdamW + CosineAnnealingLR training loop with best-checkpoint restore. Per-class classification report and confusion matrix. Grad-CAM implementation applied to final conv layer — spatial attribution maps. Pitfalls: no skip connections in deep nets, test-set checkpoint selection, Grad-CAM layer choice, class imbalance in accuracy, no weight decay.

### `object_detection.ipynb`
IoU computation. NMS algorithm implementation. Bounding box visualisation with `matplotlib.patches`. torchvision Faster R-CNN head replacement (`FastRCNNPredictor`). Train/eval mode behaviour in detection models. Mean Average Precision (mAP) implementation from precision-recall curve. Pitfalls: classification metrics for detection, NMS threshold, box format mixing, `eval()` for detection, confidence threshold tuning on test set.

### `image_segmentation.ipynb`
Full U-Net: `DoubleConv` blocks, encoder, bottleneck, `ConvTranspose2d` decoder, skip connections. Combo loss: CrossEntropyLoss + Dice loss. IoU and Dice metric computation per class. Prediction vs ground-truth visualisation (image / true mask / predicted mask grid). Pitfalls: pixel accuracy on imbalanced classes, skip connection size mismatch, CrossEntropyLoss alone for imbalanced classes, geometry augmentation without mask transform, `Upsample` vs `ConvTranspose2d`.

### `cv_pretrained_models.ipynb`
ResNet18, EfficientNet-B0, MobileNetV3-Small loaded with `weights="DEFAULT"`. Head replacement pattern for each architecture. Frozen backbone training. Architecture comparison: accuracy vs latency vs parameters. ResNet18 as fixed feature extractor — 512-dim embeddings, PCA visualisation, LogisticRegression on top. Model efficiency scatter plot. Pitfalls: `weights=None`, wrong resolution, `eval()` for feature extraction, training budget comparison, hardware latency testing.

### `cv_evaluation.ipynb`
Full classification report and confusion matrix (raw + normalised). Robustness testing: brightness, Gaussian noise, blur corruptions. Top-1 and Top-2 accuracy. Per-class one-vs-rest AUC. Most confidently wrong prediction analysis. Pitfalls: no failure mode inspection, top-1 only for ordinal classes, clean-only evaluation, overall accuracy for imbalanced data, val vs test accuracy conflation.

---

## Dependencies
```
torch, torchvision, numpy, matplotlib, sklearn, scipy, PIL
```

## Data
Simulated 64×224×224 PIL images representing ecological land-cover patches (healthy riparian, moderate, degraded; water/vegetation/soil) with pixel-level masks and bounding box annotations — all generated procedurally with known ground truth.
