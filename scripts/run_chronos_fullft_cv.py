
import os

# ============================================================
# CPU / GPU环境
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
    "expandable_segments:True"


from pathlib import Path

import argparse
import json
import shutil
import gc
import traceback
import time

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from chronos import Chronos2Pipeline

from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
)


# ============================================================
# 参数
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--trial",
    type=int,
    required=True
)

parser.add_argument(
    "--lr",
    type=float,
    required=True
)

parser.add_argument(
    "--wd",
    type=float,
    required=True
)

args = parser.parse_args()


TRIAL_ID = args.trial
LR = args.lr
WEIGHT_DECAY = args.wd


# ============================================================
# 固定配置
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_OUT_DIR = (
    ROOT
    / "release_repro/chronos_fullft/2025_work"
)

TRIAL_DIR = (
    BASE_OUT_DIR
    / f"trial_{TRIAL_ID:02d}"
)

TRIAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LOAD_PATH = (
    ROOT
    / "data" / "model_inputs" / "chronos" / "unified_load_v1.pkl.gz"
)

GFS_PATH = (
    ROOT
    / "data" / "model_inputs" / "chronos" / "chronos_gfs_15min.pkl.gz"
)


MODEL_ID = "amazon/chronos-2"

CONTEXT_DAYS = 28
POINTS_PER_DAY = 96

CONTEXT_LEN = (
    CONTEXT_DAYS
    *
    POINTS_PER_DAY
)

PRED_LEN = 96


WEATHER_COLS = [

    "temp_mean",

    "humidity_mean",

    "shortwave_mean_11city",

    "wind_speed_mean",

    "cloud_cover_mean",

    "precipitation_std",
]


MAX_STEPS = 5000

BATCH_SIZE = 28

EVAL_STEPS = 50

PATIENCE = 4

SEED = 42


# ============================================================
# NO-GAP folds
# ============================================================

FOLD_CONFIGS = {

    "Fold1": {

        "train_start":
            "2024-02-16",

        "train_end":
            "2025-03-31",

        "valid_start":
            "2025-04-01",

        "valid_end":
            "2025-06-30",
    },


    "Fold2": {

        "train_start":
            "2024-02-16",

        "train_end":
            "2025-06-30",

        "valid_start":
            "2025-07-01",

        "valid_end":
            "2025-09-30",
    },


    "Fold3": {

        "train_start":
            "2024-02-16",

        "train_end":
            "2025-09-30",

        "valid_start":
            "2025-10-01",

        "valid_end":
            "2025-12-31",
    },
}


# ============================================================
# Eval history
# ============================================================

class EvalHistoryCallback(
    TrainerCallback
):

    def __init__(self):

        self.history = []


    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        **kwargs
    ):

        if metrics is None:

            return


        loss = metrics.get(
            "eval_loss"
        )


        if loss is not None:

            self.history.append({

                "step":
                    int(
                        state.global_step
                    ),

                "eval_loss":
                    float(
                        loss
                    ),
            })


# ============================================================
# 构造 Full FT Train / Validation tasks
#
# 和 Final LoRA 数据定义完全一致
# ============================================================

def build_train_tasks(
    start_date,
    end_date,
    load_df,
    gfs_df
):

    tasks = []
    dates = []
    skipped = []


    for D in pd.date_range(
        start_date,
        end_date,
        freq="D"
    ):

        hist_start = (
            D
            -
            pd.Timedelta(
                days=CONTEXT_DAYS
            )
        )

        hist_end = (
            D
            -
            pd.Timedelta(
                days=1
            )
        )


        hist_load = (

            load_df[

                (
                    load_df["target_date"]
                    >=
                    hist_start
                )

                &

                (
                    load_df["target_date"]
                    <=
                    hist_end
                )
            ]

            .sort_values(
                "timestamp"
            )
        )


        future_load = (

            load_df[

                load_df["target_date"]
                ==
                D
            ]

            .sort_values(
                "timestamp"
            )
        )


        weather_all = (

            gfs_df[

                (
                    gfs_df["target_date"]
                    >=
                    hist_start
                )

                &

                (
                    gfs_df["target_date"]
                    <=
                    D
                )
            ]

            .sort_values(
                "timestamp"
            )
        )


        weather_future = (

            gfs_df[

                gfs_df["target_date"]
                ==
                D
            ]

            .sort_values(
                "timestamp"
            )
        )


        if len(hist_load) != CONTEXT_LEN:

            skipped.append(
                (
                    str(D.date()),
                    f"hist_load={len(hist_load)}"
                )
            )

            continue


        if len(future_load) != PRED_LEN:

            skipped.append(
                (
                    str(D.date()),
                    f"future_load={len(future_load)}"
                )
            )

            continue


        if len(weather_all) != (
            CONTEXT_LEN
            +
            PRED_LEN
        ):

            skipped.append(
                (
                    str(D.date()),
                    f"weather_all={len(weather_all)}"
                )
            )

            continue


        if len(weather_future) != PRED_LEN:

            skipped.append(
                (
                    str(D.date()),
                    f"weather_future={len(weather_future)}"
                )
            )

            continue


        # ----------------------------------------------------
        # 历史负荷
        # ----------------------------------------------------

        target_hist = (

            hist_load[
                "load_filled"
            ]

            .to_numpy(
                dtype=np.float32
            )
        )


        # ----------------------------------------------------
        # 当前目标日真实标签
        #
        # 无效标签仍保持 NaN
        # ----------------------------------------------------

        target_future = (

            future_load[
                "load_raw"
            ]

            .to_numpy(
                dtype=np.float32
            )
        )


        target = np.concatenate(
            [
                target_hist,
                target_future,
            ]
        )


        task = {

            "target":
                target,

            "past_covariates": {

                col:

                    weather_all[
                        col
                    ]

                    .to_numpy(
                        dtype=np.float32
                    )

                for col
                in WEATHER_COLS
            },

            "future_covariates": {

                col:

                    weather_future[
                        col
                    ]

                    .to_numpy(
                        dtype=np.float32
                    )

                for col
                in WEATHER_COLS
            },
        }


        tasks.append(
            task
        )

        dates.append(
            D
        )


    return (
        tasks,
        dates,
        skipped
    )


# ============================================================
# 正式推理 tasks
#
# 当前日真实负荷绝不送入模型
# ============================================================

def build_inference_tasks(
    start_date,
    end_date,
    load_df,
    gfs_df
):

    tasks = []
    dates = []

    actuals = []
    masks = []

    skipped = []


    for D in pd.date_range(
        start_date,
        end_date,
        freq="D"
    ):

        hist_start = (
            D
            -
            pd.Timedelta(
                days=CONTEXT_DAYS
            )
        )

        hist_end = (
            D
            -
            pd.Timedelta(
                days=1
            )
        )


        hist_load = (

            load_df[

                (
                    load_df["target_date"]
                    >=
                    hist_start
                )

                &

                (
                    load_df["target_date"]
                    <=
                    hist_end
                )
            ]

            .sort_values(
                "timestamp"
            )
        )


        actual_day = (

            load_df[

                load_df["target_date"]
                ==
                D
            ]

            .sort_values(
                "timestamp"
            )
        )


        hist_weather = (

            gfs_df[

                (
                    gfs_df["target_date"]
                    >=
                    hist_start
                )

                &

                (
                    gfs_df["target_date"]
                    <=
                    hist_end
                )
            ]

            .sort_values(
                "timestamp"
            )
        )


        future_weather = (

            gfs_df[

                gfs_df["target_date"]
                ==
                D
            ]

            .sort_values(
                "timestamp"
            )
        )


        if len(hist_load) != CONTEXT_LEN:

            skipped.append(
                (
                    str(D.date()),
                    "hist_load"
                )
            )

            continue


        if len(actual_day) != PRED_LEN:

            skipped.append(
                (
                    str(D.date()),
                    "actual_day"
                )
            )

            continue


        if len(hist_weather) != CONTEXT_LEN:

            skipped.append(
                (
                    str(D.date()),
                    "hist_weather"
                )
            )

            continue


        if len(future_weather) != PRED_LEN:

            skipped.append(
                (
                    str(D.date()),
                    "future_weather"
                )
            )

            continue


        task = {

            "target":

                hist_load[
                    "load_filled"
                ]

                .to_numpy(
                    dtype=np.float32
                ),


            "past_covariates": {

                col:

                    hist_weather[
                        col
                    ]

                    .to_numpy(
                        dtype=np.float32
                    )

                for col
                in WEATHER_COLS
            },


            "future_covariates": {

                col:

                    future_weather[
                        col
                    ]

                    .to_numpy(
                        dtype=np.float32
                    )

                for col
                in WEATHER_COLS
            },
        }


        tasks.append(
            task
        )

        dates.append(
            D
        )


        actuals.append(

            actual_day[
                "load_raw"
            ]

            .to_numpy(
                dtype=np.float32
            )
        )


        masks.append(

            actual_day[
                "load_valid"
            ]

            .to_numpy(
                dtype=bool
            )
        )


    return (
        tasks,
        dates,
        actuals,
        masks,
        skipped
    )


# ============================================================
# 单 Fold
# ============================================================

def run_one_fold(
    fold_name,
    load_df,
    gfs_df
):

    cfg = (
        FOLD_CONFIGS[
            fold_name
        ]
    )


    # --------------------------------------------------------
    # 每折重新固定随机种子
    # --------------------------------------------------------

    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


    print(
        "\n"
        +
        "=" * 90,
        flush=True
    )

    print(
        f"Trial {TRIAL_ID} | {fold_name}",
        flush=True
    )

    print(
        "=" * 90,
        flush=True
    )

    print(
        "LR:",
        LR,
        flush=True
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY,
        flush=True
    )

    print(
        "Batch:",
        BATCH_SIZE,
        flush=True
    )

    print(
        "Train:",
        cfg["train_start"],
        "->",
        cfg["train_end"],
        flush=True
    )

    print(
        "Validation:",
        cfg["valid_start"],
        "->",
        cfg["valid_end"],
        flush=True
    )


    # ========================================================
    # Train
    # ========================================================

    (
        train_tasks,
        train_dates,
        train_skipped,
    ) = build_train_tasks(

        pd.Timestamp(
            cfg[
                "train_start"
            ]
        ),

        pd.Timestamp(
            cfg[
                "train_end"
            ]
        ),

        load_df,
        gfs_df,
    )


    # ========================================================
    # Validation tasks
    #
    # 同时供 Early Stopping 使用
    # ========================================================

    (
        valid_train_tasks,
        valid_train_dates,
        valid_train_skipped,
    ) = build_train_tasks(

        pd.Timestamp(
            cfg[
                "valid_start"
            ]
        ),

        pd.Timestamp(
            cfg[
                "valid_end"
            ]
        ),

        load_df,
        gfs_df,
    )


    print(
        "Train tasks:",
        len(train_tasks),
        flush=True
    )

    print(
        "Validation tasks:",
        len(valid_train_tasks),
        flush=True
    )

    print(
        "Train skipped:",
        len(train_skipped),
        flush=True
    )

    print(
        "Validation skipped:",
        len(valid_train_skipped),
        flush=True
    )


    # ========================================================
    # Fold目录
    # ========================================================

    fold_dir = (
        TRIAL_DIR
        /
        fold_name.lower()
    )

    train_dir = (
        fold_dir
        /
        "trainer_tmp"
    )


    if fold_dir.exists():

        shutil.rmtree(
            fold_dir
        )


    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 每一个Fold都重新加载 fresh Chronos-2
    # ========================================================

    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        0
    )


    pipeline = (
        Chronos2Pipeline
        .from_pretrained(

            MODEL_ID,

            device_map=
                "cuda:0",

            dtype=
                torch.bfloat16,
        )
    )


    total_params = sum(

        p.numel()

        for p
        in pipeline.model.parameters()
    )


    trainable_params = sum(

        p.numel()

        for p
        in pipeline.model.parameters()

        if p.requires_grad
    )


    print(
        "Total params:",
        f"{total_params:,}",
        flush=True
    )

    print(
        "Trainable before fit:",
        f"{trainable_params:,}",
        flush=True
    )


    # ========================================================
    # Early stopping
    # ========================================================

    eval_history = (
        EvalHistoryCallback()
    )


    early_stop = (
        EarlyStoppingCallback(

            early_stopping_patience=
                PATIENCE
        )
    )


    # ========================================================
    # Full Fine-tuning
    # ========================================================

    print(
        "\nStart Full FT...",
        flush=True
    )


    start_time = (
        time.time()
    )


    fold_pipeline = pipeline.fit(

        inputs=
            train_tasks,

        validation_inputs=
            valid_train_tasks,

        prediction_length=
            PRED_LEN,

        context_length=
            CONTEXT_LEN,

        min_past=
            CONTEXT_LEN,

        # ====================================================
        # 关键：Full Fine-tuning
        # ====================================================

        finetune_mode=
            "full",

        learning_rate=
            LR,

        weight_decay=
            WEIGHT_DECAY,

        num_steps=
            MAX_STEPS,

        batch_size=
            BATCH_SIZE,

        output_dir=
            train_dir,

        finetuned_ckpt_name=
            "best_fullft",

        callbacks=[
            early_stop,
            eval_history,
        ],

        disable_data_parallel=
            True,

        dataloader_num_workers=
            0,

        logging_steps=
            10,

        eval_steps=
            EVAL_STEPS,

        save_steps=
            EVAL_STEPS,

        save_total_limit=
            1,

        disable_tqdm=
            False,

        report_to=
            "none",
    )


    torch.cuda.synchronize()


    train_seconds = (
        time.time()
        -
        start_time
    )


    # ========================================================
    # 验证曲线
    # ========================================================

    history_df = pd.DataFrame(
        eval_history.history
    )


    history_path = (
        fold_dir
        /
        "validation_eval_history.csv"
    )


    history_df.to_csv(
        history_path,
        index=False
    )


    if len(history_df) > 0:

        best_idx = (

            history_df[
                "eval_loss"
            ]

            .idxmin()
        )


        best_step = int(

            history_df.loc[
                best_idx,
                "step"
            ]
        )


        best_eval_loss = float(

            history_df.loc[
                best_idx,
                "eval_loss"
            ]
        )


        stopped_step = int(

            history_df[
                "step"
            ]
            .max()
        )


    else:

        best_step = None
        best_eval_loss = None
        stopped_step = None


    print(
        "\nBest step:",
        best_step,
        flush=True
    )

    print(
        "Best eval loss:",
        best_eval_loss,
        flush=True
    )

    print(
        "Stopped step:",
        stopped_step,
        flush=True
    )


    # ========================================================
    # 确认 Full FT 后仍然100%参数可训练
    # ========================================================

    final_total_params = sum(

        p.numel()

        for p
        in fold_pipeline.model.parameters()
    )


    final_trainable_params = sum(

        p.numel()

        for p
        in fold_pipeline.model.parameters()

        if p.requires_grad
    )


    trainable_ratio = (

        final_trainable_params

        /

        final_total_params

        *
        100
    )


    peak_vram_gb = (

        torch.cuda.max_memory_allocated(
            0
        )

        /
        1024**3
    )


    print(
        "Trainable ratio:",
        f"{trainable_ratio:.4f}%",
        flush=True
    )

    print(
        "Peak VRAM:",
        f"{peak_vram_gb:.3f} GB",
        flush=True
    )


    # ========================================================
    # 正式 Validation 推理
    # ========================================================

    (
        inference_tasks,
        valid_dates,
        actuals,
        masks,
        inference_skipped,
    ) = build_inference_tasks(

        pd.Timestamp(
            cfg[
                "valid_start"
            ]
        ),

        pd.Timestamp(
            cfg[
                "valid_end"
            ]
        ),

        load_df,
        gfs_df,
    )


    print(
        "\nInference days:",
        len(inference_tasks),
        flush=True
    )


    quantile_preds, point_preds = (
        fold_pipeline
        .predict_quantiles(

            inputs=
                inference_tasks,

            prediction_length=
                PRED_LEN,

            context_length=
                CONTEXT_LEN,

            quantile_levels=[
                0.5
            ],

            batch_size=
                BATCH_SIZE,

            cross_learning=
                False,
        )
    )


    pred_array = np.stack([

        p
        .squeeze(0)
        .cpu()
        .numpy()

        for p
        in point_preds

    ]).astype(
        np.float32
    )


    actual_array = np.stack(
        actuals
    ).astype(
        np.float32
    )


    mask_array = np.stack(
        masks
    ).astype(
        bool
    )

    # ================= FULL FT FINAL PREDICTION SAVE =================

    _final_dates = pd.to_datetime(
        valid_dates
    )

    if pred_array.shape != actual_array.shape:
        raise RuntimeError(
            f"pred_array shape {pred_array.shape} "
            f"!= actual_array shape {actual_array.shape}"
        )

    if pred_array.shape != mask_array.shape:
        raise RuntimeError(
            f"pred_array shape {pred_array.shape} "
            f"!= mask_array shape {mask_array.shape}"
        )

    if pred_array.ndim != 2:
        raise RuntimeError(
            f"预测数组应为二维 (N_day, 96)，"
            f"实际为 {pred_array.shape}"
        )

    if pred_array.shape[1] != PRED_LEN:
        raise RuntimeError(
            f"第二维应为 PRED_LEN={PRED_LEN}，"
            f"实际为 {pred_array.shape[1]}"
        )

    if len(_final_dates) != pred_array.shape[0]:
        raise RuntimeError(
            f"日期数 {len(_final_dates)} "
            f"!= 预测天数 {pred_array.shape[0]}"
        )


    _final_pred_dir = (ROOT / 'results' / 'reproduction' / 'chronos_fullft' / '2025_work' / '_predictions_raw')

    _final_pred_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # 每一天对应96个点
    _final_target_date = np.repeat(
        _final_dates.normalize().values,
        PRED_LEN
    )

    _final_point = np.tile(
        np.arange(
            1,
            PRED_LEN + 1
        ),
        len(_final_dates)
    )


    # 山西96点：
    # 1 -> 00:15
    # ...
    # 96 -> 24:00
    _final_time_one_day = []

    for _k in range(
        1,
        PRED_LEN + 1
    ):

        _minutes = _k * 15

        if _minutes == 24 * 60:

            _label = "24:00"

        else:

            _hh = _minutes // 60
            _mm = _minutes % 60

            _label = (
                f"{_hh:02d}:"
                f"{_mm:02d}"
            )

        _final_time_one_day.append(
            _label
        )


    _final_time = np.tile(
        _final_time_one_day,
        len(_final_dates)
    )


    # timestamp 使用真实连续时间：
    # target_date + 15min ... + 1440min
    _final_minutes = np.tile(
        np.arange(
            1,
            PRED_LEN + 1
        ) * 15,
        len(_final_dates)
    )

    _final_timestamp = (
        pd.to_datetime(
            _final_target_date
        )
        +
        pd.to_timedelta(
            _final_minutes,
            unit="m"
        )
    )


    _final_prediction_df = pd.DataFrame({

        "target_date":
            pd.to_datetime(
                _final_target_date
            ),

        "point":
            _final_point.astype(
                int
            ),

        "time":
            _final_time,

        "timestamp":
            _final_timestamp,

        "y_true":
            actual_array.reshape(
                -1
            ),

        "y_pred":
            pred_array.reshape(
                -1
            ),

        "valid":
            mask_array.reshape(
                -1
            ).astype(
                bool
            ),
    })


    _final_start = (
        pd.Timestamp(
            cfg["valid_start"]
        )
        .strftime(
            "%Y%m%d"
        )
    )

    _final_end = (
        pd.Timestamp(
            cfg["valid_end"]
        )
        .strftime(
            "%Y%m%d"
        )
    )


    _final_prediction_path = (
        _final_pred_dir
        /
        f"pred_{_final_start}_{_final_end}.csv"
    )


    _final_prediction_df.to_csv(
        _final_prediction_path,
        index=False
    )


    print(
        f"FINAL_PRED_SAVED: "
        f"{_final_prediction_path} | "
        f"shape={_final_prediction_df.shape}"
    )

    # ================= END FINAL PREDICTION SAVE =================



    valid = (

        mask_array

        &

        np.isfinite(
            actual_array
        )

        &

        np.isfinite(
            pred_array
        )

        &

        (
            actual_array
            >
            0
        )
    )


    y_true = (
        actual_array[
            valid
        ]
    )


    y_pred = (
        pred_array[
            valid
        ]
    )


    # ========================================================
    # Metrics
    # ========================================================

    mape = (

        np.mean(

            np.abs(
                y_pred
                -
                y_true
            )

            /

            y_true
        )

        *
        100
    )


    wmape = (

        np.sum(

            np.abs(
                y_pred
                -
                y_true
            )
        )

        /

        np.sum(
            y_true
        )

        *
        100
    )


    mae = (

        np.mean(

            np.abs(
                y_pred
                -
                y_true
            )
        )
    )


    rmse = (

        np.sqrt(

            np.mean(

                (
                    y_pred
                    -
                    y_true
                )
                ** 2
            )
        )
    )


    print(
        "\n"
        +
        "=" * 70,
        flush=True
    )

    print(
        f"{fold_name} RESULT",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    print(
        f"MAPE  = {mape:.6f}%",
        flush=True
    )

    print(
        f"WMAPE = {wmape:.6f}%",
        flush=True
    )

    print(
        f"MAE   = {mae:.3f} MW",
        flush=True
    )

    print(
        f"RMSE  = {rmse:.3f} MW",
        flush=True
    )


    # ========================================================
    # Fold结果
    # ========================================================

    result = {

        "trial":
            TRIAL_ID,

        "fold":
            fold_name,

        "learning_rate":
            LR,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,

        "max_steps":
            MAX_STEPS,

        "best_step":
            best_step,

        "stopped_step":
            stopped_step,

        "best_eval_loss":
            best_eval_loss,

        "train_tasks":
            len(train_tasks),

        "validation_tasks":
            len(inference_tasks),

        "scored_points":
            int(
                valid.sum()
            ),

        "mape":
            float(
                mape
            ),

        "wmape":
            float(
                wmape
            ),

        "mae":
            float(
                mae
            ),

        "rmse":
            float(
                rmse
            ),

        "trainable_ratio_percent":
            float(
                trainable_ratio
            ),

        "peak_vram_gb":
            float(
                peak_vram_gb
            ),

        "train_seconds":
            float(
                train_seconds
            ),
    }


    result_path = (
        fold_dir
        /
        "fold_result.json"
    )


    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


    # ========================================================
    # HPO阶段不保留大checkpoint
    #
    # 我们只保存：
    #   fold_result.json
    #   validation_eval_history.csv
    # ========================================================

    del fold_pipeline
    del pipeline

    gc.collect()

    torch.cuda.empty_cache()


    if train_dir.exists():

        shutil.rmtree(
            train_dir
        )


    return result


# ============================================================
# Trial 主程序
# ============================================================

def main():

    torch.set_num_threads(
        1
    )


    try:

        torch.set_num_interop_threads(
            1
        )

    except RuntimeError:

        pass


    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


    print(
        "\n"
        +
        "=" * 100,
        flush=True
    )

    print(
        f"FULL FT TRIAL {TRIAL_ID}",
        flush=True
    )

    print(
        "=" * 100,
        flush=True
    )

    print(
        "LR:",
        LR,
        flush=True
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY,
        flush=True
    )

    print(
        "Batch:",
        BATCH_SIZE,
        flush=True
    )

    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        flush=True
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
        flush=True
    )


    # ========================================================
    # 数据一次加载
    # ========================================================

    load_df = pd.read_pickle(
        LOAD_PATH
    )


    gfs_df = pd.read_pickle(
        GFS_PATH
    )


    load_df[
        "target_date"
    ] = (

        pd.to_datetime(

            load_df[
                "目标日期"
            ]
        )

        .dt.normalize()
    )


    load_df[
        "timestamp"
    ] = pd.to_datetime(

        load_df[
            "timestamp"
        ]
    )


    gfs_df[
        "target_date"
    ] = (

        pd.to_datetime(

            gfs_df[
                "target_date"
            ]
        )

        .dt.normalize()
    )


    gfs_df[
        "timestamp"
    ] = pd.to_datetime(

        gfs_df[
            "timestamp"
        ]
    )


    # ========================================================
    # 顺序跑3折
    # ========================================================

    results = []


    try:

        for fold_name in [

            "Fold1",
            "Fold2",
            "Fold3",

        ]:

            result = run_one_fold(

                fold_name,
                load_df,
                gfs_df
            )

            results.append(
                result
            )


        # ====================================================
        # Trial summary
        # ====================================================

        result_df = pd.DataFrame(
            results
        )


        mean_mape = float(

            result_df[
                "mape"
            ]
            .mean()
        )


        mean_wmape = float(

            result_df[
                "wmape"
            ]
            .mean()
        )


        pooled_mape = float(

            np.average(

                result_df[
                    "mape"
                ],

                weights=
                    result_df[
                        "scored_points"
                    ]
            )
        )


        summary = {

            "trial":
                TRIAL_ID,

            "learning_rate":
                LR,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "Fold1_MAPE":
                float(
                    result_df.loc[
                        result_df["fold"] == "Fold1",
                        "mape"
                    ].iloc[0]
                ),

            "Fold2_MAPE":
                float(
                    result_df.loc[
                        result_df["fold"] == "Fold2",
                        "mape"
                    ].iloc[0]
                ),

            "Fold3_MAPE":
                float(
                    result_df.loc[
                        result_df["fold"] == "Fold3",
                        "mape"
                    ].iloc[0]
                ),

            "Fold1_best_step":
                int(
                    result_df.loc[
                        result_df["fold"] == "Fold1",
                        "best_step"
                    ].iloc[0]
                ),

            "Fold2_best_step":
                int(
                    result_df.loc[
                        result_df["fold"] == "Fold2",
                        "best_step"
                    ].iloc[0]
                ),

            "Fold3_best_step":
                int(
                    result_df.loc[
                        result_df["fold"] == "Fold3",
                        "best_step"
                    ].iloc[0]
                ),

            "3Fold_Mean_MAPE":
                mean_mape,

            "3Fold_Mean_WMAPE":
                mean_wmape,

            "Pooled_MAPE":
                pooled_mape,

            "status":
                "success",
        }


        result_df.to_csv(

            TRIAL_DIR
            /
            "fold_results.csv",

            index=False
        )


        with open(

            TRIAL_DIR
            /
            "trial_summary.json",

            "w",
            encoding="utf-8"

        ) as f:

            json.dump(
                summary,
                f,
                indent=2
            )


        print(
            "\n"
            +
            "=" * 100,
            flush=True
        )

        print(
            f"TRIAL {TRIAL_ID} SUMMARY",
            flush=True
        )

        print(
            "=" * 100,
            flush=True
        )

        print(
            result_df[
                [
                    "fold",
                    "best_step",
                    "mape",
                    "wmape",
                    "peak_vram_gb",
                ]
            ]
            .to_string(
                index=False
            ),
            flush=True
        )

        print(
            f"\n3-Fold Mean MAPE = "
            f"{mean_mape:.6f}%",
            flush=True
        )


    except Exception as e:

        traceback.print_exc()


        fail_summary = {

            "trial":
                TRIAL_ID,

            "learning_rate":
                LR,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "status":
                "failed",

            "error":
                f"{type(e).__name__}: {e}",
        }


        with open(

            TRIAL_DIR
            /
            "trial_summary.json",

            "w",
            encoding="utf-8"

        ) as f:

            json.dump(
                fail_summary,
                f,
                indent=2
            )


        raise


if __name__ == "__main__":

    main()
