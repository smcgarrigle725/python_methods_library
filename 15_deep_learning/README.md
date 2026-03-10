# 15 — Deep Learning

PyTorch-based deep learning from first principles through CNNs, sequence models, transfer learning, and rigorous evaluation. Each notebook is self-contained and runnable on CPU, with GPU support where available.

---

## Notebooks

### `pytorch_fundamentals.ipynb`
Tensor operations and autograd: manual gradient computation, `.backward()`, `.grad`. `nn.Module` subclassing with `nn.Sequential`. DataLoader with shuffling and batching. Full training loop: `zero_grad` → forward → loss → `backward` → `step`. Learning rate scheduler (`ReduceLROnPlateau`). Checkpoint saving with `torch.save` including normalisation statistics. Pitfalls: forgetting `zero_grad`, train/eval mode, test-set normalisation, `.item()` before backward, `no_grad` at inference.

### `training_regularisation.ipynb`
Configurable model factory for dropout/BatchNorm combinations. L2 weight decay via `weight_decay` in Adam. Dropout rates compared empirically. StepLR, CosineAnnealingLR schedule visualisation. Custom `EarlyStopper` class with best-state restoration. Gradient clipping with `clip_grad_norm_` — before/after norm comparison. Pitfalls: `eval()` with dropout, BatchNorm after Dropout, early stopping patience too small, monitoring train loss, not restoring best weights.

### `cnns.ipynb`
2D CNN with Conv→BatchNorm→ReLU→Pool blocks. `AdaptiveAvgPool2d` for resolution-agnostic classification head. Training loop with CosineAnnealing and best-checkpoint restore. First conv filter visualisation. Feature map visualisation after block 1. 1D CNN for multivariate time series — same pattern, `Conv1d`. Pitfalls: no padding, MaxPool size miscalculation, BatchNorm placement, CrossEntropyLoss with one-hot targets, unnormalised inputs.

### `sequence_models.ipynb`
Sliding window dataset construction for time series. `nn.LSTM` and `nn.GRU` with `batch_first=True`. Shared `train_seq` function for fair comparison. Prediction vs true plot with residual analysis. Multi-step ahead recursive forecasting with error accumulation visualisation. Additive attention over LSTM outputs with attention weight plot. Pitfalls: `batch_first` default, padded sequences, long sequences without attention, multi-step uncertainty, DataLoader shuffling.

### `transfer_learning.ipynb`
Pretrained `ConvBackbone` on source task (4 classes). Feature extraction: frozen backbone, new head only. Full fine-tuning with discriminative learning rates (backbone LR < head LR). Accuracy comparison: scratch vs frozen vs fine-tuned. `torchvision.models.resnet18` head replacement and layer freezing. ImageNet normalisation constants. Pitfalls: high LR for backbone, ImageNet normalisation mismatch, immediate unfreezing, `eval()` during fine-tuning, domain-shifted transfer assumptions.

### `model_evaluation_dl.ipynb`
ROC and precision-recall curves with AUC/AP scores. Multi-seed variance analysis: mean ± SD across 5 seeds. Temperature scaling calibration with optimal T via scipy optimisation. Calibration curve before/after scaling. MC Dropout uncertainty estimation: mean and std over 50 forward passes. Most uncertain prediction identification. Pitfalls: single-seed reporting, softmax as calibrated probability, accuracy on imbalanced data, validation-set metric reporting, `eval()` mode for MC Dropout.

---

## Dependencies
```
torch, numpy, matplotlib, sklearn, scipy
torchvision (optional, for pretrained models: pip install torchvision)
```

## Data
Simulated ecological tabular data (species richness from environmental predictors), 2D spatial sensor maps (32×32 water quality grids), and multivariate water quality time series — all with known ground truth for validating model estimates.
