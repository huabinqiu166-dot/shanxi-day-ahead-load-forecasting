# Reproducibility

Before public release, the final frozen forecasting pipeline was independently
rerun.

The purpose was to verify that the final configurations remained executable
and produced consistent results.

## Components Reproduced

- LightGBM
- TFT
- Chronos-2 Zero-shot
- Chronos-2 LoRA
- Chronos-2 Full Fine-tuning
- Final Ridge Ensemble
- Two-model Ridge ablations

## Official vs Release Reproduction

2026 monthly mean MAPE:

| Model | Official | Release reproduction |
|---|---:|---:|
| LightGBM | 2.546427 | 2.546427 |
| TFT | 2.606553 | 2.606553 |
| Chronos-2 Zero-shot | 2.106621 | 2.106621 |
| Chronos-2 LoRA | 1.990602 | 1.990058 |
| Chronos-2 Full FT | 1.949120 | 1.943570 |
| Ridge Ensemble | 1.905930 | 1.901927 |

GPU-trained models may exhibit small numerical differences due to floating
point operations, CUDA kernels and optimization trajectories.

Historical official results are not replaced by the release reproduction.

The final reproduced Ridge model retained:

alpha = 100
