# Experiment History

## Baselines

1. LightGBM V14
2. TFT V22-V1

## Chronos-2 Adaptation

1. Zero-shot
2. LoRA Fine-tuning
3. Full Fine-tuning

## Final Ensemble

Base models:

- LightGBM
- TFT
- Chronos-2 Full Fine-tuning

Fusion methods evaluated during development:

- Equal Weight
- Simplex
- Ridge Stacking

The released final fusion method is Ridge Stacking.

Fusion-method and hyperparameter selection use 2025 validation only.

2026 is reserved for final testing.

Historical hyperparameter-search scripts and superseded experimental versions
are intentionally excluded from the public release.
