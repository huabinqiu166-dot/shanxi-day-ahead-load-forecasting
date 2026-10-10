# Shanxi Day-Ahead Load Forecasting

Province-level day-ahead electricity load forecasting for Shanxi, China.

The task is to use all information available through day **D-1** to forecast the complete load curve of day **D** at **96 x 15-minute** resolution.

This repository includes:

- LightGBM
- Temporal Fusion Transformer (TFT)
- Chronos-2 Zero-shot
- Chronos-2 LoRA fine-tuning
- Chronos-2 Full Fine-Tuning
- Ridge heterogeneous ensemble

The project focuses on a complete and reproducible forecasting pipeline rather than a state-of-the-art claim.

---

## Key Results

Lower MAPE is better.

### Official Historical Results

| Model | 2025 3-Fold Mean MAPE (%) | 2026 Monthly Mean MAPE (%) |
|---|---:|---:|
| LightGBM | 2.355709 | 2.546427 |
| TFT | 2.406419 | 2.606553 |
| Chronos-2 Zero-shot | 1.979280 | 2.106621 |
| Chronos-2 LoRA | 1.834057 | 1.990602 |
| Chronos-2 Full Fine-Tuning | **1.746641** | 1.949120 |
| **Final Ridge Ensemble** | - | **1.905930** |

### Ridge Ensemble Ablation

| Combination | 2026 Monthly Mean MAPE (%) |
|---|---:|
| LightGBM + TFT | 2.224383 |
| LightGBM + Chronos FullFT | 1.924466 |
| TFT + Chronos FullFT | 1.916504 |
| **LightGBM + TFT + Chronos FullFT** | **1.905930** |

The pairwise combinations are descriptive ablations.

Final Ridge hyperparameter selection uses only the 2025 chronological validation period.

---

## Evaluation Protocol

The frozen chronological validation protocol is:

| Fold | Training Period | Validation Period |
|---|---|---|
| Fold 1 | through 2025-03-31 | 2025-04-01 to 2025-06-30 |
| Fold 2 | through 2025-06-30 | 2025-07-01 to 2025-09-30 |
| Fold 3 | through 2025-09-30 | 2025-10-01 to 2025-12-31 |

No artificial gap is inserted between training and validation.

The 2026 period is reserved for final testing.

Final target date:

`2026-08-20`

---

## Models

### LightGBM

The final LightGBM model uses 48 engineered features including:

- historical load lags
- rolling load statistics
- calendar information
- holiday information
- GFS weather variables

### Temporal Fusion Transformer

Main frozen settings:

- Encoder length: 672
- Decoder length: 96
- Hidden size: 32
- Continuous hidden size: 16
- Attention heads: 4
- Dropout: 0.1
- Batch size: 16
- Median quantile: 0.5

### Chronos-2

Chronos uses:

- 28-day context
- 96-step forecast horizon
- historical system load
- six weather variables

The final Chronos experiments do not use calendar features.

Three Chronos settings are included:

- Zero-shot
- LoRA fine-tuning
- Full fine-tuning

### Final Ensemble

The final Ridge ensemble combines:

- LightGBM
- TFT
- Chronos-2 Full Fine-Tuning

---

## Public Data

Human-readable processed data:

- `data/processed/shanxi_system_load.csv`
- `data/processed/gfs_weather_15min.csv`
- `data/processed/data_manifest.json`

`shanxi_system_load.csv` contains:

- `timestamp`
- `system_load_mw`

There is one system-load variable.

`timestamp` is the time coordinate, not another load variable.

---

## Frozen Reproduction Inputs

Frozen model-ready inputs are included under:

- `data/model_inputs/lightgbm/`
- `data/model_inputs/tft/`
- `data/model_inputs/chronos/`
- `data/model_inputs/fusion/`

Large files are gzip-compressed to reduce repository size.

The compressed files were verified by exact dataframe round-trip checks.

---

## Repository Structure

    .
    ├── configs/
    ├── data/
    │   ├── processed/
    │   └── model_inputs/
    ├── docs/
    ├── results/
    │   ├── official/
    │   └── reproduction/
    ├── scripts/
    ├── src/
    ├── requirements-lgb-tft.txt
    ├── requirements-chronos.txt
    └── README.md

---

## Environment

### LightGBM / TFT / Fusion

Tested with Python 3.12.11.

    pip install -r requirements-lgb-tft.txt

### Chronos

Tested with Python 3.11.13.

The reproduction environment used PyTorch 2.5.1+cu118.

    pip install -r requirements-chronos.txt

---

## Reproduction Scripts

Main release scripts:

- `scripts/run_lightgbm.py`
- `scripts/run_tft.py`
- `scripts/run_chronos_zero_shot.py`
- `scripts/run_chronos_lora_cv.py`
- `scripts/run_chronos_lora_rolling.py`
- `scripts/summarize_chronos_lora_rolling.py`
- `scripts/run_chronos_fullft_cv.py`
- `scripts/run_chronos_fullft_rolling.py`
- `scripts/run_fusion.py`

All release scripts use repository-relative paths.

---

## Official vs Release Reproduction Results

Official historical results:

`results/official/`

Release reproduction results:

`results/reproduction/`

These two result sets are intentionally kept separate.

Fine-tuned Chronos models can show small numerical differences because GPU training is not bitwise deterministic.

---

## Data and Model-Weight Policy

This repository does not distribute:

- original commercial Excel workbooks
- seller permission screenshots
- additional seller-provided raw electricity-market variables
- Chronos base-model weights
- LoRA adapter weights
- Full Fine-Tuning checkpoints
- hyperparameter-search artifacts
- credentials or secrets
- Conda environments or caches

Only the permitted Shanxi system-load series and derived release data are included.

---

## Documentation

Additional methodological details are available in:

- `docs/data_protocol.md`
- `docs/experiment_history.md`
- `docs/reproducibility.md`

---

## License

No open-source license has been selected yet.
