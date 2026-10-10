# ============================================================
# GitHub Release Reproduction
# Step 2 — TFT V22-V1 Final
#
# 完整重新训练：
#   A. 2025 NO-GAP 3-Fold
#   B. 2026 Jan-Aug Expanding Rolling
#
# Final protocol:
# - target = load_residual_1d
# - weight = target_weight = load_valid
# - Encoder = 672
# - Decoder = 96
# - Batch = 16
# - Fixed epochs = 59
# - Seed = 42
# - prediction = lag1 + predicted residual
#
# 所有新结果只保存到 release_repro/tft
# 不覆盖任何历史正式文件
# ============================================================

from pathlib import Path
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks import TQDMProgressBar

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)

from pytorch_forecasting.data import EncoderNormalizer
from pytorch_forecasting.metrics import QuantileLoss


# ============================================================
# 0. 路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data" / "model_inputs" / "tft" / "all_v22_roll_gfs_v1.pkl.gz"
)

OUT_ROOT = (
    ROOT
    / "results" / "reproduction"
    / "tft"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Final frozen config
# ============================================================

SEED = 42
BATCH_SIZE = 16
FIXED_EPOCHS = 59

TARGET_COL = "系统负荷实际值(MW)"
RESIDUAL_COL = "load_residual_1d"
WEIGHT_COL = "target_weight"

KNOWN_CATEGORICALS = [
    "slot",
    "weekday",
    "month",
    "is_weekend",
    "holiday_type",
]

DECODER_FEATURES = [
    "dec_load_lag_1d",
    "dec_load_lag_2d",
    "dec_load_lag_3d",
    "dec_load_lag_7d",
    "dec_load_same_slot_mean_7d",
    "dec_load_lag_14d",
    "dec_residual_lag1",

    "dec_temp_mean_11city",
    "dec_humidity_mean_11city",
    "dec_shortwave_mean_11city",

    "dec_temp_change_1d",
    "dec_humidity_change_1d",
    "dec_shortwave_change_1d",

    "dec_precip_mean_11city",
    "dec_precip_std_11city",
]

assert len(KNOWN_CATEGORICALS) == 5
assert len(DECODER_FEATURES) == 15


# ============================================================
# 2. Reference results
# ============================================================

REF_2025 = {
    "Fold1": 2.708321,
    "Fold2": 2.428444,
    "Fold3": 2.082491,
}

REF_2025_MEAN = 2.406419
REF_2025_POOLED = 2.401821

REF_2026 = {
    "2026-01": 2.439636,
    "2026-02": 2.831400,
    "2026-03": 3.108214,
    "2026-04": 3.152866,
    "2026-05": 2.696875,
    "2026-06": 2.217907,
    "2026-07": 2.327746,
    "2026-08": 2.077779,
}

REF_2026_MEAN = 2.606553
REF_2026_POOLED = 2.628034
REF_2026_WMAPE = 2.548167
REF_2026_POINTS = 22271


# ============================================================
# 3. 随机种子
# ============================================================

def reset_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pl.seed_everything(
        seed,
        workers=True
    )


# ============================================================
# 4. Metrics
# ============================================================

def calc_metrics(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    err = y_true - y_pred
    ae = np.abs(err)

    return {

        "MAE":
            float(ae.mean()),

        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        err ** 2
                    )
                )
            ),

        "MAPE":
            float(
                np.mean(
                    ae / y_true
                ) * 100
            ),

        "WMAPE":
            float(
                ae.sum()
                /
                y_true.sum()
                * 100
            ),
    }


# ============================================================
# 5. 读取冻结 TFT 数据
# ============================================================

df = pd.read_pickle(
    DATA_PATH
).copy()

df["sample_id"] = (
    df["sample_id"]
    .astype(str)
)

df["sample_date"] = pd.to_datetime(
    df["sample_date"]
)

for c in KNOWN_CATEGORICALS:

    df[c] = (
        df[c]
        .astype(str)
    )


# 最终正式 supervision weight
df[WEIGHT_COL] = (
    df["load_valid"]
    .astype(float)
)


print("=" * 100)
print("TFT V22-V1 RELEASE REPRODUCTION")
print("=" * 100)

print(
    "data shape =",
    df.shape
)

print(
    "samples =",
    df["sample_id"].nunique()
)

print(
    "sample range =",
    df["sample_date"].min(),
    "→",
    df["sample_date"].max()
)


# ============================================================
# 6. 只保留至少有一个合法 Decoder 标签的 sample
# ============================================================

valid_per_sample = (

    df[
        df["part"] == "decoder"
    ]

    .groupby(
        "sample_id"
    )["load_valid"]

    .sum()
)


GOOD_IDS = set(

    valid_per_sample[
        valid_per_sample > 0
    ].index
)


print(
    "supervised samples =",
    len(GOOD_IDS)
)


# ============================================================
# 7. Dataset builder
# ============================================================

def build_datasets(
    train_raw,
    monitor_raw,
    test_raw,
):

    reset_seed(SEED)

    training_ds = TimeSeriesDataSet(

        train_raw,

        time_idx=
            "series_time_idx",

        target=
            RESIDUAL_COL,

        group_ids=[
            "sample_id"
        ],

        weight=
            WEIGHT_COL,

        min_encoder_length=
            672,

        max_encoder_length=
            672,

        min_prediction_length=
            96,

        max_prediction_length=
            96,

        static_categoricals=[],
        static_reals=[],

        time_varying_known_categoricals=
            KNOWN_CATEGORICALS,

        time_varying_known_reals=
            DECODER_FEATURES,

        time_varying_unknown_categoricals=[],

        time_varying_unknown_reals=[
            TARGET_COL,
            RESIDUAL_COL,
        ],

        target_normalizer=
            EncoderNormalizer(
                method="standard"
            ),

        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,

        randomize_length=False,
    )


    monitor_ds = (
        TimeSeriesDataSet
        .from_dataset(

            training_ds,
            monitor_raw,

            predict=True,
            stop_randomization=True,
        )
    )


    test_ds = (
        TimeSeriesDataSet
        .from_dataset(

            training_ds,
            test_raw,

            predict=True,
            stop_randomization=True,
        )
    )


    return (
        training_ds,
        monitor_ds,
        test_ds,
    )


# ============================================================
# 8. Model builder
# ============================================================

def create_model(training_ds):

    reset_seed(SEED)

    model = (

        TemporalFusionTransformer
        .from_dataset(

            training_ds,

            learning_rate=
                0.001,

            hidden_size=
                32,

            hidden_continuous_size=
                16,

            attention_head_size=
                4,

            dropout=
                0.1,

            lstm_layers=
                1,

            output_size=
                1,

            loss=
                QuantileLoss(
                    quantiles=[0.5]
                ),

            log_interval=
                -1,

            log_val_interval=
                -1,

            reduce_on_plateau_patience=
                4,
        )
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    assert n_params == 130608, (
        f"参数量异常：{n_params}"
    )

    return model


# ============================================================
# 9. Trainer builder
# ============================================================

def create_trainer():

    # 不需要 ipywidgets 也能运行
    progress_bar = TQDMProgressBar(
        refresh_rate=5
    )

    reset_seed(SEED)

    trainer = pl.Trainer(

        max_epochs=
            FIXED_EPOCHS,

        accelerator=(
            "gpu"
            if torch.cuda.is_available()
            else "cpu"
        ),

        devices=1,

        gradient_clip_val=
            0.1,

        logger=False,

        enable_checkpointing=False,

        enable_model_summary=False,

        num_sanity_val_steps=0,

        log_every_n_steps=5,

        enable_progress_bar=True,

        callbacks=[
            progress_bar
        ],
    )

    return trainer


# ============================================================
# 10. Prediction → load
# ============================================================

def predict_and_score(
    model,
    test_loader,
    test_raw,
    expected_samples,
    expected_valid_points,
):

    model.eval()

    pred_obj = model.predict(

        test_loader,

        mode="prediction",

        return_index=True,

        trainer_kwargs={

            "accelerator": (
                "gpu"
                if torch.cuda.is_available()
                else "cpu"
            ),

            "devices": 1,

            "logger": False,

            "enable_progress_bar": False,
        }
    )


    pred_residual = (

        pred_obj.output
        .detach()
        .cpu()
        .numpy()
    )


    pred_residual = np.squeeze(
        pred_residual
    )


    if pred_residual.ndim == 1:

        pred_residual = (
            pred_residual
            .reshape(
                1,
                -1
            )
        )


    assert (
        pred_residual.shape
        ==
        (
            expected_samples,
            96
        )
    )


    pred_ids = (

        pred_obj.index[
            "sample_id"
        ]

        .astype(str)
        .to_numpy()
    )


    pred_long = pd.DataFrame({

        "sample_id":
            np.repeat(
                pred_ids,
                96
            ),

        "step":
            np.tile(
                np.arange(96),
                len(pred_ids)
            ),

        "pred_residual":
            pred_residual
            .reshape(-1),
    })


    truth = (

        test_raw[
            test_raw["part"]
            ==
            "decoder"
        ]

        .copy()

        .sort_values(
            [
                "sample_id",
                "series_time_idx"
            ]
        )

        .reset_index(
            drop=True
        )
    )


    truth["step"] = (

        truth
        .groupby(
            "sample_id"
        )
        .cumcount()
    )


    assert (
        len(truth)
        ==
        expected_samples
        *
        96
    )


    compare = (

        truth[
            [
                "sample_id",
                "sample_date",
                "series_time_idx",
                "step",
                TARGET_COL,
                "dec_load_lag_1d",
                "load_valid",
            ]
        ]

        .merge(

            pred_long,

            on=[
                "sample_id",
                "step"
            ],

            how="inner",

            validate="one_to_one",
        )
    )


    # TFT预测 residual
    # 最终负荷 = 昨日同点负荷 + residual
    compare[
        "pred_load"
    ] = (

        compare[
            "dec_load_lag_1d"
        ]

        +

        compare[
            "pred_residual"
        ]
    )


    # 正式评价只看真实合法点
    score = (

        compare[
            compare["load_valid"]
            ==
            1
        ]

        .copy()
    )


    assert (
        len(score)
        ==
        expected_valid_points
    )


    metrics = calc_metrics(

        score[TARGET_COL],

        score["pred_load"],
    )


    del pred_obj

    return (
        compare,
        score,
        metrics,
    )


# ============================================================
# 11. 2025 3-Fold plans
# ============================================================

FOLDS = [

    {
        "fold": "Fold1",

        "val_start":
            "2025-04-01",

        "val_end":
            "2025-07-01",

        "monitor_start":
            "2025-03-01",

        "monitor_end":
            "2025-04-01",

        "expected_train":
            437,

        "expected_monitor":
            31,

        "expected_eval":
            88,

        "expected_valid_points":
            8372,
    },

    {
        "fold": "Fold2",

        "val_start":
            "2025-07-01",

        "val_end":
            "2025-10-01",

        "monitor_start":
            "2025-06-01",

        "monitor_end":
            "2025-07-01",

        "expected_train":
            525,

        "expected_monitor":
            29,

        "expected_eval":
            92,

        "expected_valid_points":
            8832,
    },

    {
        "fold": "Fold3",

        "val_start":
            "2025-10-01",

        "val_end":
            "2026-01-01",

        "monitor_start":
            "2025-09-01",

        "monitor_end":
            "2025-10-01",

        "expected_train":
            617,

        "expected_monitor":
            30,

        "expected_eval":
            92,

        "expected_valid_points":
            8772,
    },
]


# ============================================================
# 12. 运行一个 Fold
# ============================================================

def run_fold(cfg):

    fold = cfg["fold"]

    val_start = pd.Timestamp(
        cfg["val_start"]
    )

    val_end = pd.Timestamp(
        cfg["val_end"]
    )

    monitor_start = pd.Timestamp(
        cfg["monitor_start"]
    )

    monitor_end = pd.Timestamp(
        cfg["monitor_end"]
    )


    print("\n\n" + "#" * 100)
    print(f"2025 {fold}")
    print("#" * 100)


    train_raw = df[

        (
            df["sample_date"]
            <
            val_start
        )

        &

        (
            df["sample_id"]
            .isin(GOOD_IDS)
        )

    ].copy()


    monitor_raw = train_raw[

        (
            train_raw["sample_date"]
            >=
            monitor_start
        )

        &

        (
            train_raw["sample_date"]
            <
            monitor_end
        )

    ].copy()


    eval_raw = df[

        (
            df["sample_date"]
            >=
            val_start
        )

        &

        (
            df["sample_date"]
            <
            val_end
        )

        &

        (
            df["sample_id"]
            .isin(GOOD_IDS)
        )

    ].copy()


    n_train = (
        train_raw["sample_id"]
        .nunique()
    )

    n_monitor = (
        monitor_raw["sample_id"]
        .nunique()
    )

    n_eval = (
        eval_raw["sample_id"]
        .nunique()
    )


    print(
        "Train / Monitor / Eval =",
        n_train,
        "/",
        n_monitor,
        "/",
        n_eval,
    )


    assert n_train == cfg["expected_train"]
    assert n_monitor == cfg["expected_monitor"]
    assert n_eval == cfg["expected_eval"]


    (
        training_ds,
        monitor_ds,
        eval_ds,
    ) = build_datasets(

        train_raw,
        monitor_raw,
        eval_raw,
    )


    assert len(training_ds) == n_train
    assert len(monitor_ds) == n_monitor
    assert len(eval_ds) == n_eval


    reset_seed(SEED)

    train_loader = (
        training_ds
        .to_dataloader(

            train=True,

            batch_size=
                BATCH_SIZE,

            num_workers=0,
        )
    )


    monitor_loader = (
        monitor_ds
        .to_dataloader(

            train=False,

            batch_size=
                BATCH_SIZE,

            num_workers=0,
        )
    )


    eval_loader = (
        eval_ds
        .to_dataloader(

            train=False,

            batch_size=
                BATCH_SIZE,

            num_workers=0,
        )
    )


    model = create_model(
        training_ds
    )


    trainer = create_trainer()


    start = time.time()


    trainer.fit(

        model,

        train_dataloaders=
            train_loader,

        val_dataloaders=
            monitor_loader,
    )


    elapsed = (
        time.time()
        -
        start
    )


    fold_dir = (
        OUT_ROOT
        /
        "2025"
        /
        fold.lower()
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    ckpt_path = (
        fold_dir
        /
        f"tft_v22_release_{fold.lower()}_epoch59.ckpt"
    )


    trainer.save_checkpoint(
        str(ckpt_path)
    )


    (
        compare,
        score,
        metrics,
    ) = predict_and_score(

        model,
        eval_loader,
        eval_raw,

        expected_samples=
            n_eval,

        expected_valid_points=
            cfg["expected_valid_points"],
    )


    compare["fold"] = fold


    compare.to_csv(

        fold_dir
        /
        f"{fold.lower()}_predictions.csv",

        index=False,
    )


    print(
        f"\n{fold}"
        f" | MAPE={metrics['MAPE']:.6f}%"
        f" | WMAPE={metrics['WMAPE']:.6f}%"
        f" | scored={len(score)}"
        f" | time={elapsed/60:.2f} min"
    )


    result = {

        "Fold":
            fold,

        "Train_Samples":
            n_train,

        "Monitor_Samples":
            n_monitor,

        "Eval_Samples":
            n_eval,

        "Scored_Points":
            len(score),

        **metrics,

        "Train_Minutes":
            elapsed / 60,
    }


    del model
    del trainer
    del training_ds
    del monitor_ds
    del eval_ds

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return (
        result,
        compare,
    )


# ============================================================
# 13. 正式运行 2025 三折
# ============================================================

cv_results = []
cv_predictions = []


for cfg in FOLDS:

    result, pred = run_fold(
        cfg
    )

    cv_results.append(
        result
    )

    cv_predictions.append(
        pred
    )


cv_summary = pd.DataFrame(
    cv_results
)

cv_all = pd.concat(
    cv_predictions,
    ignore_index=True
)


cv_score = (

    cv_all[
        cv_all["load_valid"]
        ==
        1
    ]

    .copy()
)


assert len(cv_score) == 25976


cv_pooled = calc_metrics(

    cv_score[TARGET_COL],

    cv_score["pred_load"],
)


cv_mean = float(
    cv_summary["MAPE"].mean()
)


print("\n\n" + "=" * 100)
print("2025 TFT 3-FOLD RELEASE CHECK")
print("=" * 100)


for _, row in cv_summary.iterrows():

    ref = REF_2025[
        row["Fold"]
    ]

    print(
        f"{row['Fold']}: "
        f"reference={ref:.6f}% | "
        f"reproduced={row['MAPE']:.6f}% | "
        f"diff={row['MAPE']-ref:+.6f} pp"
    )


print(
    f"\n3-Fold Mean:"
    f" reference={REF_2025_MEAN:.6f}%"
    f" | reproduced={cv_mean:.6f}%"
    f" | diff={cv_mean-REF_2025_MEAN:+.6f} pp"
)

print(
    f"Pooled MAPE:"
    f" reference={REF_2025_POOLED:.6f}%"
    f" | reproduced={cv_pooled['MAPE']:.6f}%"
)


cv_summary.to_csv(
    OUT_ROOT
    / "tft_2025_cv_summary.csv",
    index=False
)

cv_all.to_csv(
    OUT_ROOT
    / "tft_2025_predictions.csv",
    index=False
)


# ============================================================
# 14. 2026 Rolling plans
# ============================================================

ROLLING = [

    {
        "month": "2026-01",
        "test_start": "2026-01-01",
        "test_end": "2026-02-01",
        "monitor_start": "2025-12-01",
        "monitor_end": "2026-01-01",
        "expected_train": 709,
        "expected_monitor": 31,
        "expected_test": 31,
        "expected_points": 2976,
    },

    {
        "month": "2026-02",
        "test_start": "2026-02-01",
        "test_end": "2026-03-01",
        "monitor_start": "2026-01-01",
        "monitor_end": "2026-02-01",
        "expected_train": 740,
        "expected_monitor": 31,
        "expected_test": 28,
        "expected_points": 2688,
    },

    {
        "month": "2026-03",
        "test_start": "2026-03-01",
        "test_end": "2026-04-01",
        "monitor_start": "2026-02-01",
        "monitor_end": "2026-03-01",
        "expected_train": 768,
        "expected_monitor": 28,
        "expected_test": 31,
        "expected_points": 2976,
    },

    {
        "month": "2026-04",
        "test_start": "2026-04-01",
        "test_end": "2026-05-01",
        "monitor_start": "2026-03-01",
        "monitor_end": "2026-04-01",
        "expected_train": 799,
        "expected_monitor": 31,
        "expected_test": 30,
        "expected_points": 2880,
    },

    {
        "month": "2026-05",
        "test_start": "2026-05-01",
        "test_end": "2026-06-01",
        "monitor_start": "2026-04-01",
        "monitor_end": "2026-05-01",
        "expected_train": 829,
        "expected_monitor": 30,
        "expected_test": 31,
        "expected_points": 2975,
    },

    {
        "month": "2026-06",
        "test_start": "2026-06-01",
        "test_end": "2026-07-01",
        "monitor_start": "2026-05-01",
        "monitor_end": "2026-06-01",
        "expected_train": 860,
        "expected_monitor": 31,
        "expected_test": 30,
        "expected_points": 2880,
    },

    {
        "month": "2026-07",
        "test_start": "2026-07-01",
        "test_end": "2026-08-01",
        "monitor_start": "2026-06-01",
        "monitor_end": "2026-07-01",
        "expected_train": 890,
        "expected_monitor": 30,
        "expected_test": 31,
        "expected_points": 2976,
    },

    {
        "month": "2026-08",
        "test_start": "2026-08-01",
        "test_end": "2026-08-21",
        "monitor_start": "2026-07-01",
        "monitor_end": "2026-08-01",
        "expected_train": 921,
        "expected_monitor": 31,
        "expected_test": 20,
        "expected_points": 1920,
    },
]


# ============================================================
# 15. 运行一个 rolling month
# ============================================================

def run_month(cfg):

    month = cfg["month"]

    test_start = pd.Timestamp(
        cfg["test_start"]
    )

    test_end = pd.Timestamp(
        cfg["test_end"]
    )

    monitor_start = pd.Timestamp(
        cfg["monitor_start"]
    )

    monitor_end = pd.Timestamp(
        cfg["monitor_end"]
    )


    print("\n\n" + "#" * 100)
    print(month)
    print("#" * 100)


    train_raw = df[

        (
            df["sample_date"]
            <
            test_start
        )

        &

        (
            df["sample_id"]
            .isin(GOOD_IDS)
        )

    ].copy()


    monitor_raw = train_raw[

        (
            train_raw["sample_date"]
            >=
            monitor_start
        )

        &

        (
            train_raw["sample_date"]
            <
            monitor_end
        )

    ].copy()


    test_raw = df[

        (
            df["sample_date"]
            >=
            test_start
        )

        &

        (
            df["sample_date"]
            <
            test_end
        )

        &

        (
            df["sample_id"]
            .isin(GOOD_IDS)
        )

    ].copy()


    n_train = (
        train_raw["sample_id"]
        .nunique()
    )

    n_monitor = (
        monitor_raw["sample_id"]
        .nunique()
    )

    n_test = (
        test_raw["sample_id"]
        .nunique()
    )


    assert n_train == cfg["expected_train"]
    assert n_monitor == cfg["expected_monitor"]
    assert n_test == cfg["expected_test"]


    print(
        "Train / Monitor / Test =",
        n_train,
        "/",
        n_monitor,
        "/",
        n_test,
    )


    (
        training_ds,
        monitor_ds,
        test_ds,
    ) = build_datasets(

        train_raw,
        monitor_raw,
        test_raw,
    )


    reset_seed(SEED)

    train_loader = (
        training_ds
        .to_dataloader(
            train=True,
            batch_size=BATCH_SIZE,
            num_workers=0,
        )
    )

    monitor_loader = (
        monitor_ds
        .to_dataloader(
            train=False,
            batch_size=BATCH_SIZE,
            num_workers=0,
        )
    )

    test_loader = (
        test_ds
        .to_dataloader(
            train=False,
            batch_size=BATCH_SIZE,
            num_workers=0,
        )
    )


    model = create_model(
        training_ds
    )

    trainer = create_trainer()


    start = time.time()


    trainer.fit(

        model,

        train_dataloaders=
            train_loader,

        val_dataloaders=
            monitor_loader,
    )


    elapsed = (
        time.time()
        -
        start
    )


    month_dir = (
        OUT_ROOT
        /
        "2026"
        /
        month
    )

    month_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    trainer.save_checkpoint(

        str(
            month_dir
            /
            f"tft_v22_release_{month}_epoch59.ckpt"
        )
    )


    (
        compare,
        score,
        metrics,
    ) = predict_and_score(

        model,
        test_loader,
        test_raw,

        expected_samples=
            n_test,

        expected_valid_points=
            cfg["expected_points"],
    )


    compare["month"] = month


    compare.to_csv(

        month_dir
        /
        "predictions.csv",

        index=False,
    )


    print(
        f"{month}"
        f" | MAPE={metrics['MAPE']:.6f}%"
        f" | WMAPE={metrics['WMAPE']:.6f}%"
        f" | scored={len(score)}"
        f" | time={elapsed/60:.2f} min"
    )


    result = {

        "Month":
            month,

        "Train_Samples":
            n_train,

        "Monitor_Samples":
            n_monitor,

        "Test_Days":
            n_test,

        "Valid_Points":
            len(score),

        **metrics,

        "Train_Minutes":
            elapsed / 60,
    }


    del model
    del trainer
    del training_ds
    del monitor_ds
    del test_ds

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return (
        result,
        compare,
    )


# ============================================================
# 16. 正式运行 2026 Jan-Aug
# ============================================================

monthly_results = []
monthly_predictions = []


for cfg in ROLLING:

    result, pred = run_month(
        cfg
    )

    monthly_results.append(
        result
    )

    monthly_predictions.append(
        pred
    )


monthly_summary = pd.DataFrame(
    monthly_results
)


all_2026 = (

    pd.concat(
        monthly_predictions,
        ignore_index=True
    )

    .sort_values(
        [
            "sample_date",
            "series_time_idx"
        ]
    )

    .reset_index(
        drop=True
    )
)


score_2026 = (

    all_2026[
        all_2026["load_valid"]
        ==
        1
    ]

    .copy()
)


assert len(score_2026) == 22271


pooled_2026 = calc_metrics(

    score_2026[TARGET_COL],

    score_2026["pred_load"],
)


monthly_mean = float(
    monthly_summary[
        "MAPE"
    ].mean()
)


# ============================================================
# 17. 保存
# ============================================================

monthly_summary.to_csv(
    OUT_ROOT
    / "tft_2026_monthly.csv",
    index=False
)

all_2026.to_csv(
    OUT_ROOT
    / "tft_2026_predictions.csv",
    index=False
)


# ============================================================
# 18. Release reproduction final check
# ============================================================

print("\n\n" + "=" * 100)
print("TFT RELEASE REPRODUCTION FINAL CHECK")
print("=" * 100)


print("\n2025:")

for _, row in cv_summary.iterrows():

    ref = REF_2025[
        row["Fold"]
    ]

    print(
        f"{row['Fold']:5s}"
        f" ref={ref:.6f}%"
        f" | new={row['MAPE']:.6f}%"
        f" | diff={row['MAPE']-ref:+.6f} pp"
    )


print(
    f"\n2025 Mean:"
    f" ref={REF_2025_MEAN:.6f}%"
    f" | new={cv_mean:.6f}%"
    f" | diff={cv_mean-REF_2025_MEAN:+.6f} pp"
)


print("\n2026:")

for _, row in monthly_summary.iterrows():

    ref = REF_2026[
        row["Month"]
    ]

    print(
        f"{row['Month']}"
        f" ref={ref:.6f}%"
        f" | new={row['MAPE']:.6f}%"
        f" | diff={row['MAPE']-ref:+.6f} pp"
    )


print(
    f"\n2026 Monthly Mean:"
    f" ref={REF_2026_MEAN:.6f}%"
    f" | new={monthly_mean:.6f}%"
    f" | diff={monthly_mean-REF_2026_MEAN:+.6f} pp"
)

print(
    f"2026 Pooled MAPE:"
    f" ref={REF_2026_POOLED:.6f}%"
    f" | new={pooled_2026['MAPE']:.6f}%"
)

print(
    f"2026 Pooled WMAPE:"
    f" ref={REF_2026_WMAPE:.6f}%"
    f" | new={pooled_2026['WMAPE']:.6f}%"
)

print(
    f"2026 scored points:"
    f" ref={REF_2026_POINTS}"
    f" | new={len(score_2026)}"
)


summary_json = {

    "model":
        "TFT V22-V1 Final",

    "seed":
        SEED,

    "batch_size":
        BATCH_SIZE,

    "fixed_epochs":
        FIXED_EPOCHS,

    "encoder_length":
        672,

    "prediction_length":
        96,

    "weight":
        "target_weight = load_valid",

    "2025_fold_mape": {
        row["Fold"]:
            float(row["MAPE"])
        for _, row
        in cv_summary.iterrows()
    },

    "2025_mean_mape":
        cv_mean,

    "2025_pooled_mape":
        pooled_cv["MAPE"],

    "2026_monthly_mape": {
        row["Month"]:
            float(row["MAPE"])
        for _, row
        in monthly_summary.iterrows()
    },

    "2026_monthly_mean_mape":
        monthly_mean,

    "2026_pooled_mape":
        pooled_2026["MAPE"],

    "2026_pooled_wmape":
        pooled_2026["WMAPE"],

    "2026_scored_points":
        len(score_2026),
}


with open(
    OUT_ROOT
    / "reproduction_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_json,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\nSaved to:")
print(OUT_ROOT)

print("\n✅ TFT RELEASE REPRODUCTION FINISHED")