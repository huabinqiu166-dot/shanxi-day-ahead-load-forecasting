
import os

# ============================================================
# CPU限制
#
# 一定放在numpy / torch import之前
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


# ============================================================
# Imports
# ============================================================

from pathlib import Path

import argparse
import json
import shutil
import gc
import time

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from chronos import Chronos2Pipeline

from transformers import TrainerCallback


# ============================================================
# 2. 路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


OUT_DIR = (
    ROOT
    / "release_repro/chronos_lora/2026_rolling"
)


LOAD_PATH = (
    ROOT
    / "data" / "model_inputs" / "chronos" / "unified_load_v1.pkl.gz"
)


GFS_PATH = (
    ROOT
    / "data" / "model_inputs" / "chronos" / "chronos_gfs_15min.pkl.gz"
)


# ============================================================
# 3. 最终冻结配置
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


# ------------------------------------------------------------
# V3 scheduler horizon
# ------------------------------------------------------------

MAX_STEPS = 5000


# ------------------------------------------------------------
# 2025 NO-GAP Fold3选出的正式停止点
# ------------------------------------------------------------

FIXED_STOP_STEP = 1400


BATCH_SIZE = 28

LR = 2.833485198833818e-05


LORA_CONFIG = {

    "r":
        4,

    "lora_alpha":
        16,

    "lora_dropout":
        0.1,

    "target_modules": [

        "self_attention.q",
        "self_attention.k",
        "self_attention.v",
        "self_attention.o",
        "output_patch_embedding.output_layer",
    ],
}


# ============================================================
# 4. 2026 Rolling计划
# ============================================================

ROLLING_PLAN = {

    "2026-01": {

        "train_end":
            "2025-12-31",

        "test_start":
            "2026-01-01",

        "test_end":
            "2026-01-31",
    },


    "2026-02": {

        "train_end":
            "2026-01-31",

        "test_start":
            "2026-02-01",

        "test_end":
            "2026-02-28",
    },


    "2026-03": {

        "train_end":
            "2026-02-28",

        "test_start":
            "2026-03-01",

        "test_end":
            "2026-03-31",
    },


    "2026-04": {

        "train_end":
            "2026-03-31",

        "test_start":
            "2026-04-01",

        "test_end":
            "2026-04-30",
    },


    "2026-05": {

        "train_end":
            "2026-04-30",

        "test_start":
            "2026-05-01",

        "test_end":
            "2026-05-31",
    },


    "2026-06": {

        "train_end":
            "2026-05-31",

        "test_start":
            "2026-06-01",

        "test_end":
            "2026-06-30",
    },


    "2026-07": {

        "train_end":
            "2026-06-30",

        "test_start":
            "2026-07-01",

        "test_end":
            "2026-07-31",
    },


    "2026-08": {

        "train_end":
            "2026-07-31",

        "test_start":
            "2026-08-01",

        # 数据只到8月20日
        "test_end":
            "2026-08-20",
    },
}


# ============================================================
# 5. 固定step停止
#
# 不EarlyStopping
#
# 训练到2800以后直接停
# 此时返回的pipeline就是step1400权重
# ============================================================

class StopAtStepCallback(
    TrainerCallback
):

    def __init__(
        self,
        stop_step
    ):

        self.stop_step = int(
            stop_step
        )


    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        if (
            state.global_step
            >=
            self.stop_step
        ):

            control.should_training_stop = True


        return control


# ============================================================
# 6. 构造LoRA训练任务
#
# 对训练日期D：
#
# target =
#
#   历史28天 load_filled
#   +
#   D日 load_raw
#
# shape:
#
#   2688 + 96 = 2784
#
#
# 坏标签保持NaN
#
# 不人为插值目标标签
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
        # 历史负荷
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
        # 当前训练目标日负荷
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
        # 历史28天 + 当前目标日天气
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


        # ----------------------------------------------------
        # 当前目标日未来天气
        # ----------------------------------------------------

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
            CONTEXT_LEN + PRED_LEN
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
        # target历史部分
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
        # target未来96点标签
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
                target_future
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
# 7. 构造预测任务
#
# 非常重要：
#
# 当月真实负荷绝不会进入model input。
#
# target只有过去28天：
#
# shape = 2688
#
# 当前月actual仅用于最后评分。
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


        # ----------------------------------------------------
        # 历史28天负荷
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
        # 当前目标日actual
        #
        # 只留给评分
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 过去28天天气
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 当前目标日未来天气
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 完整性
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

            # ------------------------------------------------
            # ONLY历史负荷
            # ------------------------------------------------

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
# 8. 单个月完整流程
# ============================================================

def run_one_month(
    month_name,
    load_df,
    gfs_df
):

    cfg = (
        ROLLING_PLAN[
            month_name
        ]
    )


    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    month_tag = (
        month_name.replace(
            "-",
            "_"
        )
    )


    month_dir = (

        OUT_DIR

        /

        month_tag
    )


    result_path = (

        month_dir

        /

        "month_result.json"
    )


    pred_path = (

        month_dir

        /

        "predictions.csv"
    )


    # ========================================================
    # Resume功能
    #
    # 如果这个月已经完整跑完，
    # 重新执行大cell时不重复训练。
    # ========================================================

    if (
        result_path.exists()
        and
        pred_path.exists()
    ):

        print(
            "\n"
            +
            "=" * 86,
            flush=True
        )

        print(
            f"{month_name} 已完成，自动跳过",
            flush=True
        )

        print(
            "=" * 86,
            flush=True
        )

        return


    # --------------------------------------------------------
    # 不完整旧目录删除
    # --------------------------------------------------------

    if month_dir.exists():

        shutil.rmtree(
            month_dir
        )


    month_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # 每个月完全相同seed
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
        "=" * 86,
        flush=True
    )


    print(
        f"{month_name} Rolling",
        flush=True
    )


    print(
        "=" * 86,
        flush=True
    )


    print(
        "Train:",
        "2024-02-16",
        "->",
        cfg[
            "train_end"
        ],
        flush=True
    )


    print(
        "Test :",
        cfg[
            "test_start"
        ],
        "->",
        cfg[
            "test_end"
        ],
        flush=True
    )


    print(
        "Scheduler horizon:",
        MAX_STEPS,
        flush=True
    )


    print(
        "Fixed stop step:",
        FIXED_STOP_STEP,
        flush=True
    )


    # ========================================================
    # 9. 构造训练数据
    # ========================================================

    (
        train_tasks,
        train_dates,
        train_skipped,
    ) = build_lora_tasks(

        pd.Timestamp(
            "2024-02-16"
        ),

        pd.Timestamp(
            cfg[
                "train_end"
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


    print(
        "\n========== TRAIN DATA ==========",
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
        "Train valid labels:",
        int(
            train_valid_labels
        ),
        flush=True
    )


    print(
        "Skipped:",
        len(
            train_skipped
        ),
        flush=True
    )


    # ========================================================
    # 10. Fresh原始Chronos-2
    # ========================================================

    print(
        "\nLoading fresh amazon/chronos-2...",
        flush=True
    )


    base_pipeline = (

        Chronos2Pipeline
        .from_pretrained(

            "amazon/chronos-2",

            device_map="cuda:0",

            dtype=torch.bfloat16,
        )
    )


    # ========================================================
    # 11. 固定停止2800
    # ========================================================

    stop_callback = (

        StopAtStepCallback(
            FIXED_STOP_STEP
        )
    )


    start_time = time.time()


    print(
        "\n========== TRAINING ==========",
        flush=True
    )


    print(
        f"{month_name}: "
        f"step 0 -> {FIXED_STOP_STEP}",
        flush=True
    )


    # --------------------------------------------------------
    # 关键：
    #
    # num_steps仍然是5000
    #
    # callback让它在2800停
    #
    # 所以scheduler轨迹保持V3
    # --------------------------------------------------------

    month_pipeline = (

        base_pipeline.fit(

            inputs=
                train_tasks,

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
                month_dir,

            finetuned_ckpt_name=
                "final_lora_step1400",

            callbacks=[
                stop_callback
            ],

            disable_data_parallel=True,

            dataloader_num_workers=0,

            logging_steps=50,

            disable_tqdm=False,

            report_to="none",
        )
    )


    train_seconds = (
        time.time()
        -
        start_time
    )


    print(
        f"\n✅ {month_name} "
        f"training stopped at step "
        f"{FIXED_STOP_STEP}",
        flush=True
    )


    print(
        f"Training time = "
        f"{train_seconds / 60:.2f} min",
        flush=True
    )


    print(
        "\n接下来直接使用"
        "step1400当前权重预测。",
        flush=True
    )


    # ========================================================
    # 12. 当前月推理数据
    # ========================================================

    (
        test_tasks,
        test_dates,
        actuals,
        masks,
        test_skipped,
    ) = build_inference_tasks(

        pd.Timestamp(
            cfg[
                "test_start"
            ]
        ),

        pd.Timestamp(
            cfg[
                "test_end"
            ]
        ),

        load_df,
        gfs_df,
    )


    expected_valid_points = sum(

        mask.sum()

        for mask
        in masks
    )


    print(
        "\n========== TEST DATA ==========",
        flush=True
    )


    print(
        "Test days:",
        len(
            test_tasks
        ),
        flush=True
    )


    print(
        "Valid points:",
        int(
            expected_valid_points
        ),
        flush=True
    )


    print(
        "Skipped:",
        len(
            test_skipped
        ),
        flush=True
    )


    # ========================================================
    # 13. 推理
    # ========================================================

    expected_batches = int(

        np.ceil(

            len(
                test_tasks
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
            f"{month_name} predict"
    )


    quantile_preds, point_preds = (

        month_pipeline
        .predict_quantiles(

            inputs=
                test_tasks,

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
                lambda *args, **kwargs:
                    pbar.update(1),
        )
    )


    pbar.close()


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
        "Pred  :",
        pred_array.shape,
        flush=True
    )


    print(
        "Actual:",
        actual_array.shape,
        flush=True
    )


    print(
        "Mask  :",
        mask_array.shape,
        flush=True
    )


    # ========================================================
    # 14. 正式评分mask
    # ========================================================

    score_mask = (

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
            score_mask
        ]
    )


    y_pred = (

        pred_array[
            score_mask
        ]
    )


    # ========================================================
    # 15. Metrics
    # ========================================================

    mae = (

        np.mean(

            np.abs(
                y_pred
                -
                y_true
            )
        )
    )


    rmse = np.sqrt(

        np.mean(

            (
                y_pred
                -
                y_true
            )
            **
            2
        )
    )


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
        "\n"
        +
        "=" * 86,
        flush=True
    )


    print(
        f"{month_name} RESULT",
        flush=True
    )


    print(
        "=" * 86,
        flush=True
    )


    print(
        "Scored points:",
        int(
            score_mask.sum()
        ),
        flush=True
    )


    print(
        f"MAE   = "
        f"{mae:.3f} MW",
        flush=True
    )


    print(
        f"RMSE  = "
        f"{rmse:.3f} MW",
        flush=True
    )


    print(
        f"MAPE  = "
        f"{mape:.6f}%",
        flush=True
    )


    print(
        f"WMAPE = "
        f"{wmape:.6f}%",
        flush=True
    )


    # ========================================================
    # 16. 保存逐点预测
    # ========================================================

    rows = []


    for i, D in enumerate(
        test_dates
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

            y = (
                actual_array[
                    i,
                    j
                ]
            )


            p = (
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

                np.isfinite(y)

                and

                y > 0

                and

                np.isfinite(p)
            ):

                ape = (

                    abs(
                        p - y
                    )

                    /

                    y

                    *
                    100
                )


            else:

                ape = np.nan


            rows.append({

                "month":
                    month_name,

                "train_end":
                    cfg[
                        "train_end"
                    ],

                "fixed_stop_step":
                    FIXED_STOP_STEP,

                "target_date":
                    D,

                "timestamp":
                    timestamps[j],

                "slot":
                    j + 1,

                "load_actual":
                    y,

                "load_valid":
                    int(
                        is_valid
                    ),

                "pred_chronos_lora_v3_nogap_rolling":
                    p,

                "APE_percent":
                    ape,
            })


    pred_df = pd.DataFrame(
        rows
    )


    pred_df.to_csv(
        pred_path,
        index=False
    )


    # ========================================================
    # 17. 保存该月JSON
    # ========================================================

    result = {

        "month":
            month_name,

        "train_start":
            "2024-02-16",

        "train_end":
            cfg[
                "train_end"
            ],

        "test_start":
            cfg[
                "test_start"
            ],

        "test_end":
            cfg[
                "test_end"
            ],

        "scheduler_max_steps":
            MAX_STEPS,

        "fixed_stop_step":
            FIXED_STOP_STEP,

        "learning_rate":
            LR,

        "batch_size":
            BATCH_SIZE,

        "train_tasks":
            len(
                train_tasks
            ),

        "train_valid_labels":
            int(
                train_valid_labels
            ),

        "test_days":
            len(
                test_tasks
            ),

        "scored_points":
            int(
                score_mask.sum()
            ),

        "mae":
            float(
                mae
            ),

        "rmse":
            float(
                rmse
            ),

        "mape":
            float(
                mape
            ),

        "wmape":
            float(
                wmape
            ),

        "train_seconds":
            float(
                train_seconds
            ),

        "prediction_path":
            str(
                pred_path
            ),
    }


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
        "\nSaved prediction:",
        pred_path,
        flush=True
    )


    print(
        "Saved result:",
        result_path,
        flush=True
    )


    # ========================================================
    # 18. 清理该月GPU
    # ========================================================

    del month_pipeline

    del base_pipeline

    del train_tasks

    del test_tasks

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# 19. Worker main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(

        "--months",

        nargs="+",

        required=True,

        choices=[

            "2026-01",
            "2026-02",
            "2026-03",
            "2026-04",
            "2026-05",
            "2026-06",
            "2026-07",
            "2026-08",
        ]
    )


    args = (
        parser.parse_args()
    )


    # --------------------------------------------------------
    # CPU严格1线程
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
        "\n"
        +
        "=" * 90,
        flush=True
    )


    print(
        "2026 Rolling Worker START",
        flush=True
    )


    print(
        "Months:",
        args.months,
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
        "=" * 90,
        flush=True
    )


    # ========================================================
    # 数据每个worker只加载一次
    # ========================================================

    load_df = (
        pd.read_pickle(
            LOAD_PATH
        )
    )


    gfs_df = (
        pd.read_pickle(
            GFS_PATH
        )
    )


    # --------------------------------------------------------
    # load target_date
    # --------------------------------------------------------

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
    ] = (

        pd.to_datetime(

            load_df[
                "timestamp"
            ]
        )
    )


    # --------------------------------------------------------
    # GFS
    # --------------------------------------------------------

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
    ] = (

        pd.to_datetime(

            gfs_df[
                "timestamp"
            ]
        )
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


    # ========================================================
    # 顺序跑该GPU负责的月份
    # ========================================================

    for month_name in args.months:

        run_one_month(

            month_name,

            load_df,

            gfs_df
        )


    print(
        "\n"
        +
        "=" * 90,
        flush=True
    )


    print(
        "✅ Worker completed:",
        args.months,
        flush=True
    )


    print(
        "=" * 90,
        flush=True
    )


if __name__ == "__main__":

    main()
