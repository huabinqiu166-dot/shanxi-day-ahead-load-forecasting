
import os

# ============================================================
# CPU限制
# 必须放在 numpy / torch import 以前
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


from pathlib import Path

import argparse
import json
import shutil
import gc

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
# 固定路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_OUT_DIR = (ROOT / 'results' / 'reproduction' / 'chronos_lora' / '2025_cv')

LOAD_PATH = (
    ROOT
    / "data" / "model_inputs" / "chronos" / "unified_load_v1.pkl.gz"
)

GFS_PATH = (
    ROOT
    / "data" / "model_inputs" / "chronos" / "chronos_gfs_15min.pkl.gz"
)


# ============================================================
# 固定 Chronos 配置
# ============================================================

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
LR = 2.833485198833818e-05
EVAL_STEPS = 50

PATIENCE = 4


LORA_CONFIG = {

    "r": 4,

    "lora_alpha": 16,

    "lora_dropout": 0.1,

    "target_modules": [

        "self_attention.q",
        "self_attention.k",
        "self_attention.v",
        "self_attention.o",
        "output_patch_embedding.output_layer",
    ],
}


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
# 记录每次 eval_loss
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
                    float(loss),
            })


# ============================================================
# 构造 Train / Validation tasks
#
# Training task:
#
# target:
#   过去28天 load_filled
#   +
#   当前目标日 load_raw
#
# shape:
#   2688 + 96 = 2784
#
# 当前日坏标签保持 NaN
#
# past_covariates:
#   过去28天 + 当前日天气
#
# future_covariates:
#   当前日96点天气
# ============================================================

def build_lora_tasks(
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


        # ----------------------------------------------------
        # 过去28天历史负荷
        # ----------------------------------------------------

        hist_load = (

            load_df[

                (
                    load_df[
                        "target_date"
                    ]
                    >=
                    hist_start
                )

                &

                (
                    load_df[
                        "target_date"
                    ]
                    <=
                    hist_end
                )
            ]

            .sort_values(
                "timestamp"
            )
        )


        # ----------------------------------------------------
        # 当前目标日真实负荷
        # ----------------------------------------------------

        future_load = (

            load_df[

                load_df[
                    "target_date"
                ]
                ==
                D
            ]

            .sort_values(
                "timestamp"
            )
        )


        # ----------------------------------------------------
        # 天气：
        #
        # historical 28d + target day
        # ----------------------------------------------------

        weather_all = (

            gfs_df[

                (
                    gfs_df[
                        "target_date"
                    ]
                    >=
                    hist_start
                )

                &

                (
                    gfs_df[
                        "target_date"
                    ]
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

                gfs_df[
                    "target_date"
                ]
                ==
                D
            ]

            .sort_values(
                "timestamp"
            )
        )


        # ----------------------------------------------------
        # 完整性检查
        # ----------------------------------------------------

        if (
            len(hist_load)
            !=
            CONTEXT_LEN
        ):

            skipped.append(
                (
                    str(
                        D.date()
                    ),

                    f"hist_load={len(hist_load)}"
                )
            )

            continue


        if (
            len(future_load)
            !=
            PRED_LEN
        ):

            skipped.append(
                (
                    str(
                        D.date()
                    ),

                    f"future_load={len(future_load)}"
                )
            )

            continue


        if (
            len(weather_all)
            !=
            CONTEXT_LEN
            +
            PRED_LEN
        ):

            skipped.append(
                (
                    str(
                        D.date()
                    ),

                    f"weather_all={len(weather_all)}"
                )
            )

            continue


        if (
            len(weather_future)
            !=
            PRED_LEN
        ):

            skipped.append(
                (
                    str(
                        D.date()
                    ),

                    f"weather_future={len(weather_future)}"
                )
            )

            continue


        # ----------------------------------------------------
        # target
        # ----------------------------------------------------

        target_hist = (

            hist_load[
                "load_filled"
            ]

            .to_numpy(
                dtype=np.float32
            )
        )


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


        # ----------------------------------------------------
        # task
        # ----------------------------------------------------

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
# 正式预测 tasks
#
# 这里绝对不给未来真实负荷
#
# target shape:
#   2688
#
# future load:
#   只单独留作评分
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
                    load_df[
                        "target_date"
                    ]
                    >=
                    hist_start
                )

                &

                (
                    load_df[
                        "target_date"
                    ]
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

                load_df[
                    "target_date"
                ]
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
                    gfs_df[
                        "target_date"
                    ]
                    >=
                    hist_start
                )

                &

                (
                    gfs_df[
                        "target_date"
                    ]
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

                gfs_df[
                    "target_date"
                ]
                ==
                D
            ]

            .sort_values(
                "timestamp"
            )
        )


        if (
            len(hist_load)
            !=
            CONTEXT_LEN
        ):

            skipped.append(
                (
                    str(
                        D.date()
                    ),

                    "hist_load"
                )
            )

            continue


        if (
            len(actual_day)
            !=
            PRED_LEN
        ):

            skipped.append(
                (
                    str(
                        D.date()
                    ),

                    "actual_day"
                )
            )

            continue


        if (
            len(hist_weather)
            !=
            CONTEXT_LEN
        ):

            skipped.append(
                (
                    str(
                        D.date()
                    ),

                    "hist_weather"
                )
            )

            continue


        if (
            len(future_weather)
            !=
            PRED_LEN
        ):

            skipped.append(
                (
                    str(
                        D.date()
                    ),

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
    # 每一折都独立固定 seed
    # --------------------------------------------------------

    torch.manual_seed(
        42
    )

    np.random.seed(
        42
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            42
        )


    print(
        "\n"
        +
        "=" * 82,
        flush=True
    )

    print(
        f"{fold_name} — NO-GAP V3",
        flush=True
    )

    print(
        "=" * 82,
        flush=True
    )


    print(
        "Train:",
        cfg[
            "train_start"
        ],
        "->",
        cfg[
            "train_end"
        ],
        flush=True
    )


    print(
        "Validation:",
        cfg[
            "valid_start"
        ],
        "->",
        cfg[
            "valid_end"
        ],
        flush=True
    )


    # --------------------------------------------------------
    # Train tasks
    # --------------------------------------------------------

    (
        train_tasks,
        train_dates,
        train_skipped,
    ) = build_lora_tasks(

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


    # --------------------------------------------------------
    # Validation tasks
    #
    # 注意：
    # 这就是后面正式评分的三个月
    # --------------------------------------------------------

    (
        valid_train_tasks,
        valid_train_dates,
        valid_train_skipped,
    ) = build_lora_tasks(

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


    train_valid_labels = sum(

        np.isfinite(

            task[
                "target"
            ][
                -PRED_LEN:
            ]
        ).sum()

        for task
        in train_tasks
    )


    validation_valid_labels = sum(

        np.isfinite(

            task[
                "target"
            ][
                -PRED_LEN:
            ]
        ).sum()

        for task
        in valid_train_tasks
    )


    print(
        "\n========== Data ==========",
        flush=True
    )


    print(
        "Train tasks:",
        len(
            train_tasks
        ),
        flush=True
    )


    print(
        "Validation tasks:",
        len(
            valid_train_tasks
        ),
        flush=True
    )


    print(
        "Train valid labels:",
        train_valid_labels,
        flush=True
    )


    print(
        "Validation valid labels:",
        validation_valid_labels,
        flush=True
    )


    print(
        "Train skipped:",
        len(
            train_skipped
        ),
        flush=True
    )


    print(
        "Validation skipped:",
        len(
            valid_train_skipped
        ),
        flush=True
    )


    # --------------------------------------------------------
    # output dir
    # --------------------------------------------------------

    fold_dir = (

        BASE_OUT_DIR

        /

        fold_name.lower()
    )


    if fold_dir.exists():

        shutil.rmtree(
            fold_dir
        )


    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # 每一折都重新从原始 Chronos-2 开始
    # --------------------------------------------------------

    print(
        "\nLoading fresh amazon/chronos-2...",
        flush=True
    )


    pipeline = (
        Chronos2Pipeline
        .from_pretrained(

            "amazon/chronos-2",

            device_map="cuda:0",

            dtype=torch.bfloat16,
        )
    )


    print(
        "Model device:",
        pipeline.model.device,
        flush=True
    )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    eval_history = (
        EvalHistoryCallback()
    )


    early_stop = (
        EarlyStoppingCallback(

            early_stopping_patience=
                PATIENCE
        )
    )


    print(
        "\n========== Training ==========",
        flush=True
    )


    print(
        "Max steps:",
        MAX_STEPS,
        flush=True
    )


    print(
        "Batch size:",
        BATCH_SIZE,
        flush=True
    )


    print(
        "LR:",
        LR,
        flush=True
    )


    print(
        "Eval every:",
        EVAL_STEPS,
        flush=True
    )


    print(
        "Patience:",
        PATIENCE,
        flush=True
    )


    # --------------------------------------------------------
    # Fit
    #
    # validation_inputs 就是正式三个月验证区间
    # --------------------------------------------------------

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

        finetune_mode=
            "lora",

        lora_config=
            LORA_CONFIG,

        learning_rate=
            LR,

        num_steps=
            MAX_STEPS,

        batch_size=
            BATCH_SIZE,

        output_dir=
            fold_dir,

        finetuned_ckpt_name=
            "best_lora",

        callbacks=[
            early_stop,
            eval_history,
        ],

        disable_data_parallel=True,

        dataloader_num_workers=0,

        logging_steps=10,

        eval_steps=
            EVAL_STEPS,

        save_steps=
            EVAL_STEPS,

        disable_tqdm=False,

        report_to="none",
    )


    print(
        f"\n✅ {fold_name} training completed",
        flush=True
    )


    # --------------------------------------------------------
    # 保存 validation loss
    # --------------------------------------------------------

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


    if len(
        history_df
    ) > 0:

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
        "\n========== Early Stop ==========",
        flush=True
    )


    print(
        "Best step:",
        best_step,
        flush=True
    )


    print(
        "Best validation loss:",
        best_eval_loss,
        flush=True
    )


    print(
        "Stopped at:",
        stopped_step,
        flush=True
    )


    # ========================================================
    # 正式预测
    #
    # 日期与 EarlyStopping validation完全相同
    #
    # 但推理输入绝对不给当前日真实负荷
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
        "\n========== Validation Prediction ==========",
        flush=True
    )


    print(
        "Days:",
        len(
            inference_tasks
        ),
        flush=True
    )


    print(
        "Skipped:",
        len(
            inference_skipped
        ),
        flush=True
    )


    print(
        "Valid points:",
        sum(
            x.sum()
            for x
            in masks
        ),
        flush=True
    )


    # --------------------------------------------------------
    # progress bar
    # --------------------------------------------------------

    expected_batches = int(

        np.ceil(

            len(
                inference_tasks
            )

            *

            (
                1
                +
                len(
                    WEATHER_COLS
                )
            )

            /

            BATCH_SIZE
        )
    )


    pbar = tqdm(

        total=
            expected_batches,

        desc=
            f"{fold_name} prediction"
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

            cross_learning=False,

            after_batch=
                lambda:
                    pbar.update(1),
        )
    )


    pbar.close()


    # --------------------------------------------------------
    # array
    # --------------------------------------------------------

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


    print(
        "\nShapes:",
        flush=True
    )


    print(
        "Prediction:",
        pred_array.shape,
        flush=True
    )


    print(
        "Actual:",
        actual_array.shape,
        flush=True
    )


    print(
        "Mask:",
        mask_array.shape,
        flush=True
    )


    # --------------------------------------------------------
    # valid scoring mask
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    print(
        f"\n========== {fold_name} RESULT ==========",
        flush=True
    )


    print(
        "Scored points:",
        int(
            valid.sum()
        ),
        flush=True
    )


    print(
        f"MAPE : {mape:.6f}%",
        flush=True
    )


    print(
        f"WMAPE: {wmape:.6f}%",
        flush=True
    )


    # ========================================================
    # 保存逐点预测
    # ========================================================

    rows = []


    for i, D in enumerate(
        valid_dates
    ):

        timestamps = (

            D

            +

            pd.to_timedelta(

                np.arange(
                    1,
                    PRED_LEN + 1
                )

                *
                15,

                unit="min"
            )
        )


        for j in range(
            PRED_LEN
        ):

            actual = (
                actual_array[
                    i,
                    j
                ]
            )


            pred = (
                pred_array[
                    i,
                    j
                ]
            )


            is_valid = bool(

                mask_array[
                    i,
                    j
                ]
            )


            if (

                is_valid

                and

                np.isfinite(
                    actual
                )

                and

                actual > 0

                and

                np.isfinite(
                    pred
                )
            ):

                ape = (

                    abs(
                        pred
                        -
                        actual
                    )

                    /

                    actual

                    *
                    100
                )


            else:

                ape = np.nan


            rows.append({

                "fold":
                    fold_name,

                "target_date":
                    D,

                "timestamp":
                    timestamps[j],

                "slot":
                    j + 1,

                "load_actual":
                    actual,

                "load_valid":
                    int(
                        is_valid
                    ),

                "pred_lora_v3_nogap":
                    pred,

                "APE_percent":
                    ape,
            })


    pred_df = pd.DataFrame(
        rows
    )


    prediction_path = (

        fold_dir

        /

        f"{fold_name.lower()}_validation_predictions.csv"
    )


    pred_df.to_csv(
        prediction_path,
        index=False
    )


    # ========================================================
    # 保存结果 JSON
    # ========================================================

    result = {

        "fold":
            fold_name,

        "protocol":
            "no_gap_validation_early_stopping",

        "train_start":
            cfg[
                "train_start"
            ],

        "train_end":
            cfg[
                "train_end"
            ],

        "valid_start":
            cfg[
                "valid_start"
            ],

        "valid_end":
            cfg[
                "valid_end"
            ],

        "max_steps":
            MAX_STEPS,

        "best_step":
            best_step,

        "best_eval_loss":
            best_eval_loss,

        "stopped_step":
            stopped_step,

        "train_tasks":
            len(
                train_tasks
            ),

        "validation_tasks":
            len(
                inference_tasks
            ),

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

        "prediction_path":
            str(
                prediction_path
            ),
    }


    result_path = (

        fold_dir

        /

        "fold_result.json"
    )


    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


    print(
        "\nSaved:",
        prediction_path,
        flush=True
    )


    print(
        "Result:",
        result_path,
        flush=True
    )


    # ========================================================
    # 清显存
    # ========================================================

    del fold_pipeline

    del pipeline

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# main
#
# --folds Fold1 Fold2
#
# 或：
#
# --folds Fold3
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--folds",

        nargs="+",

        required=True,

        choices=[
            "Fold1",
            "Fold2",
            "Fold3",
        ]
    )


    args = parser.parse_args()


    # --------------------------------------------------------
    # CPU threads
    # --------------------------------------------------------

    torch.set_num_threads(
        1
    )


    try:

        torch.set_num_interop_threads(
            1
        )

    except RuntimeError:

        pass


    print(
        "\n==============================================",
        flush=True
    )

    print(
        "Process started",
        flush=True
    )

    print(
        "Folds:",
        args.folds,
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
        "Visible GPU count:",
        torch.cuda.device_count(),
        flush=True
    )

    print(
        "Visible GPU:",
        torch.cuda.get_device_name(
            0
        ),
        flush=True
    )

    print(
        "CPU threads:",
        torch.get_num_threads(),
        flush=True
    )

    print(
        "==============================================",
        flush=True
    )


    # --------------------------------------------------------
    # 数据只在该进程加载一次
    # --------------------------------------------------------

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


    print(
        "\nLoad shape:",
        load_df.shape,
        flush=True
    )


    print(
        "GFS shape:",
        gfs_df.shape,
        flush=True
    )


    # --------------------------------------------------------
    # 顺序执行本进程负责的 folds
    # --------------------------------------------------------

    for fold_name in args.folds:

        run_one_fold(

            fold_name,

            load_df,

            gfs_df
        )


    print(
        "\n✅ Process completed:",
        args.folds,
        flush=True
    )


if __name__ == "__main__":

    main()
