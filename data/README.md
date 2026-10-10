# Data

## processed/

Human-readable public processed data:

- shanxi_system_load.csv
- gfs_weather_15min.csv

shanxi_system_load.csv contains:

- timestamp
- system_load_mw

There is one system-load series. timestamp is the time coordinate.

## model_inputs/

Frozen model-ready data used by the release.

LightGBM:
- lightgbm_v14_valid_supervised_v1.pkl.gz
- lightgbm_v14_master_v1.pkl.gz
- lightgbm_v14_feature_header.csv
- lightgbm_v14_features.json

TFT:
- all_v22_roll_gfs_v1.pkl.gz

Chronos:
- unified_load_v1.pkl.gz
- chronos_gfs_15min.pkl.gz

Fusion:
- aligned_2025_three_models_and_fusions.csv.gz
- aligned_2026_three_models_and_fusions.csv.gz

The compressed files were verified by exact dataframe round-trip comparison.

Original commercial Excel workbooks and other seller-provided raw electricity-market variables are not distributed.
