# Shanxi Day-Ahead Load Forecasting

Province-level day-ahead electricity load forecasting for Shanxi, China.

## Task

Use all information available through day D-1 to forecast the complete load curve of day D at 96 x 15-minute resolution.

## Models

- LightGBM
- Temporal Fusion Transformer (TFT)
- Chronos-2 Zero-shot
- Chronos-2 LoRA
- Chronos-2 Full Fine-Tuning
- Ridge heterogeneous ensemble

## Evaluation protocol

2025 chronological validation:

- Fold 1: train through 2025-03-31, validate 2025-04-01 to 2025-06-30
- Fold 2: train through 2025-06-30, validate 2025-07-01 to 2025-09-30
- Fold 3: train through 2025-09-30, validate 2025-10-01 to 2025-12-31

2026 is reserved for final testing.

Final target date: 2026-08-20.

## Public base data

- data/processed/shanxi_system_load.csv
- data/processed/gfs_weather_15min.csv

The load CSV contains one load series:

- timestamp
- system_load_mw

## Frozen reproduction inputs

Exact model-ready inputs are included under:

- data/model_inputs/lightgbm/
- data/model_inputs/tft/
- data/model_inputs/chronos/
- data/model_inputs/fusion/

Large inputs are gzip compressed. Compression was verified by exact round-trip checks.

## Scripts

- scripts/run_lightgbm.py
- scripts/run_tft.py
- scripts/run_chronos_zero_shot.py
- scripts/run_chronos_lora_cv.py
- scripts/run_chronos_lora_rolling.py
- scripts/summarize_chronos_lora_rolling.py
- scripts/run_chronos_fullft_cv.py
- scripts/run_chronos_fullft_rolling.py
- scripts/run_fusion.py

All release scripts use repository-relative paths.

## Environments

LightGBM / TFT:

    pip install -r requirements-lgb-tft.txt

Chronos:

    pip install -r requirements-chronos.txt

Two tested environments were used:

- LightGBM/TFT: Python 3.12.11
- Chronos: Python 3.11.13

## Results

Official historical results:

    results/official/

Release reproduction results:

    results/reproduction/

Small differences can occur for GPU fine-tuning because training is not bitwise deterministic.

## Data policy

This repository does not include:

- original seller Excel workbooks
- permission screenshots
- Chronos base weights
- LoRA adapters
- FullFT checkpoints
- HPO trial files
- credentials or secrets
- Conda environments or caches

Only the permitted Shanxi system-load series and derived reproduction data are included.
