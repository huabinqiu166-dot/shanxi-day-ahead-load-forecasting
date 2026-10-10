# ============================================================
# GitHub Release Reproduction
# Step 3 — Chronos-2 Zero-shot Final
#
# Final frozen version:
#   Chronos-2 120M
#   Zero-shot
#   Context = 28d = 2688
#   Prediction = 96
#   Historical load = load_filled
#   Historical weather = 6 GFS variables
#   Future weather = same 6 GFS variables
#   cross_learning = False
#   Calendar = No
#
# 覆盖：
#   2025 Fold1 / Fold2 / Fold3
#   2026 Jan-Aug
#
# 新结果只保存到：
#   release_repro/chronos_zero_shot
#
# 不复用旧预测，不覆盖历史结果
# ============================================================

from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline


# ============================================================
# 0. 路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

LOAD_PATH = (
    ROOT
    / "data" / "model_inputs" / "chronos" / "unified_load_v1.pkl.gz"
)

WEATHER_PATH = (
    ROOT
    / "data" / "model_inputs" / "chronos" / "chronos_gfs_15min.pkl.gz"
)

OLD_PRED_PATH = (ROOT / "data" / "optional_reference" / "chronos_zero_shot_historical_predictions.csv")

OUT_DIR = (
    ROOT
    / "results" / "reproduction"
    / "chronos_zero_shot"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NEW_PRED_PATH = (
    OUT_DIR
    / "chronos_zero_shot_28d_6weather_predictions.csv"
)


# ============================================================
# 1. Final frozen config
# ============================================================

MODEL_ID = "amazon/chronos-2"

CONTEXT_DAYS = 28

POINTS_PER_DAY = 96

CONTEXT_LEN = (
    CONTEXT_DAYS
    *
    POINTS_PER_DAY
)

PRED_LEN = 96

BATCH_SIZE = 32

CHUNK_DAYS = 16


WEATHER_COLS = [

    "temp_mean",

    "humidity_mean",

    "shortwave_mean_11city",

    "wind_speed_mean",

    "cloud_cover_mean",

    "precipitation_std",
]


assert CONTEXT_LEN == 2688
assert len(WEATHER_COLS) == 6


# ============================================================
# 2. Reference
# ============================================================

REF_2025 = {

    "Fold1":
        2.056838,

    "Fold2":
        1.961519,

    "Fold3":
        1.919483,
}

REF_2025_MEAN = 1.979280

REF_2025_POOLED = 1.978045

REF_2026_MONTHLY_MEAN = 2.106621

REF_2026_POOLED = 2.117369

REF_2026_POINTS = 22271


# ============================================================
# 3. 指标
# ============================================================

def calc_metrics(
    y,
    p
):

    y = np.asarray(
        y,
        dtype=float
    )

    p = np.asarray(
        p,
        dtype=float
    )

    ae = np.abs(
        y - p
    )

    return {

        "MAE":
            float(
                ae.mean()
            ),

        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        (y - p) ** 2
                    )
                )
            ),

        "MAPE":
            float(
                np.mean(
                    ae / y
                ) * 100
            ),

        "WMAPE":
            float(
                ae.sum()
                /
                y.sum()
                * 100
            ),
    }


# ============================================================
# 4. 读取统一负荷
# ============================================================

load = pd.read_pickle(
    LOAD_PATH
).copy()


load["timestamp"] = pd.to_datetime(
    load["timestamp"]
)


load = (

    load

    .sort_values(
        "timestamp"
    )

    .reset_index(
        drop=True
    )
)


required_load_cols = [

    "timestamp",

    "load_raw",

    "load_filled",

    "load_valid",
]


for c in required_load_cols:

    assert c in load.columns, (
        f"负荷缺少列：{c}"
    )


assert not load[
    "timestamp"
].duplicated().any()


print("=" * 105)
print("CHRONOS-2 ZERO-SHOT RELEASE REPRODUCTION")
print("=" * 105)

print(
    "Load range =",
    load["timestamp"].min(),
    "→",
    load["timestamp"].max()
)

print(
    "Load rows =",
    len(load)
)


# ============================================================
# 5. 读取 GFS
# ============================================================

weather = pd.read_pickle(
    WEATHER_PATH
).copy()


weather["timestamp"] = pd.to_datetime(
    weather["timestamp"]
)


weather = (

    weather

    .sort_values(
        "timestamp"
    )

    .reset_index(
        drop=True
    )
)


for c in WEATHER_COLS:

    assert c in weather.columns, (
        f"GFS缺少列：{c}"
    )


assert not weather[
    "timestamp"
].duplicated().any()


print(
    "Weather range =",
    weather["timestamp"].min(),
    "→",
    weather["timestamp"].max()
)

print(
    "Weather variables =",
    WEATHER_COLS
)


# ============================================================
# 6. 索引
# ============================================================

load_idx = (
    load
    .set_index(
        "timestamp"
    )
)


weather_idx = (
    weather
    .set_index(
        "timestamp"
    )
)


# ============================================================
# 7. 正式评价日期
# ============================================================

fold1_dates = pd.date_range(

    "2025-04-01",

    "2025-06-30",

    freq="D"
)


fold2_dates = pd.date_range(

    "2025-07-01",

    "2025-09-30",

    freq="D"
)


fold3_dates = pd.date_range(

    "2025-10-01",

    "2025-12-31",

    freq="D"
)


dates_2026 = pd.date_range(

    "2026-01-01",

    "2026-08-20",

    freq="D"
)


TARGET_DATES = (

    fold1_dates

    .append(
        fold2_dates
    )

    .append(
        fold3_dates
    )

    .append(
        dates_2026
    )
)


assert len(
    TARGET_DATES
) == 507


TARGET_IDS = [

    d.strftime(
        "%Y-%m-%d"
    )

    for d
    in TARGET_DATES
]


print(
    "Target days =",
    len(TARGET_DATES)
)


# ============================================================
# 8. 真实标签
# ============================================================

truth = load[
    [
        "timestamp",
        "load_raw",
        "load_valid",
    ]
].copy()


# 00:15 ~ 次日00:00
# 都归属于目标日 D
truth[
    "target_date"
] = (

    truth[
        "timestamp"
    ]

    -

    pd.Timedelta(
        minutes=15
    )

).dt.normalize()


truth = truth[

    truth[
        "target_date"
    ].isin(
        TARGET_DATES
    )

].copy()


assert (
    len(truth)
    ==
    507 * 96
)


# ============================================================
# 9. 构造 28d Context + Future weather
# ============================================================

context_parts = []

future_parts = []


for i, D in enumerate(
    TARGET_DATES,
    start=1
):

    item_id = (
        D.strftime(
            "%Y-%m-%d"
        )
    )


    forecast_start = (

        D

        +

        pd.Timedelta(
            minutes=15
        )
    )


    # --------------------------------------------------------
    # 过去28天：2688个15min点
    # --------------------------------------------------------

    hist_times = pd.date_range(

        end=(
            forecast_start

            -

            pd.Timedelta(
                minutes=15
            )
        ),

        periods=
            CONTEXT_LEN,

        freq=
            "15min",
    )


    # --------------------------------------------------------
    # 目标日未来96点
    # --------------------------------------------------------

    future_times = pd.date_range(

        start=
            forecast_start,

        periods=
            PRED_LEN,

        freq=
            "15min",
    )


    # --------------------------------------------------------
    # Historical load
    # --------------------------------------------------------

    hist_load = (
        load_idx
        .reindex(
            hist_times
        )
    )


    if (
        hist_load[
            "load_filled"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(

            f"{item_id}: "
            f"历史 load_filled 有缺失"
        )


    # --------------------------------------------------------
    # Historical weather
    # --------------------------------------------------------

    hist_weather = (
        weather_idx
        .reindex(
            hist_times
        )
    )


    if (
        hist_weather[
            WEATHER_COLS
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(

            f"{item_id}: "
            f"历史天气有缺失"
        )


    # --------------------------------------------------------
    # Future weather
    # --------------------------------------------------------

    future_weather = (
        weather_idx
        .reindex(
            future_times
        )
    )


    if (
        future_weather[
            WEATHER_COLS
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(

            f"{item_id}: "
            f"未来天气有缺失"
        )


    # --------------------------------------------------------
    # context_df
    # --------------------------------------------------------

    ctx = pd.DataFrame({

        "item_id":
            item_id,

        "timestamp":
            hist_times,

        "target":
            hist_load[
                "load_filled"
            ].to_numpy(),
    })


    for c in WEATHER_COLS:

        ctx[c] = (

            hist_weather[
                c
            ]

            .to_numpy()
        )


    context_parts.append(
        ctx
    )


    # --------------------------------------------------------
    # future_df
    # --------------------------------------------------------

    fut = pd.DataFrame({

        "item_id":
            item_id,

        "timestamp":
            future_times,
    })


    for c in WEATHER_COLS:

        fut[c] = (

            future_weather[
                c
            ]

            .to_numpy()
        )


    future_parts.append(
        fut
    )


    if (
        i % 100 == 0
        or
        i == len(TARGET_DATES)
    ):

        print(
            f"Build data: "
            f"{i}/{len(TARGET_DATES)}"
        )


context_df = pd.concat(

    context_parts,

    ignore_index=True
)


future_df = pd.concat(

    future_parts,

    ignore_index=True
)


print(
    "\ncontext_df shape =",
    context_df.shape
)

print(
    "future_df shape  =",
    future_df.shape
)


assert (
    len(context_df)
    ==
    507
    *
    CONTEXT_LEN
)


assert (
    len(future_df)
    ==
    507
    *
    PRED_LEN
)


# ============================================================
# 10. 加载基础 Chronos-2
# ============================================================

print(
    "\nLoading Chronos-2 ..."
)


torch.cuda.empty_cache()


pipeline = (

    Chronos2Pipeline

    .from_pretrained(

        MODEL_ID,

        device_map=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
)


print(
    "✅ Chronos-2 loaded"
)


# ============================================================
# 11. Zero-shot prediction
# ============================================================

all_pred_parts = []


num_chunks = int(

    np.ceil(

        len(TARGET_IDS)

        /

        CHUNK_DAYS
    )
)


for chunk_i, start in enumerate(

    range(
        0,
        len(TARGET_IDS),
        CHUNK_DAYS
    ),

    start=1
):

    ids = TARGET_IDS[
        start:
        start + CHUNK_DAYS
    ]


    ctx_chunk = context_df[

        context_df[
            "item_id"
        ].isin(
            ids
        )

    ].copy()


    fut_chunk = future_df[

        future_df[
            "item_id"
        ].isin(
            ids
        )

    ].copy()


    print(

        f"Predict "
        f"[{chunk_i}/{num_chunks}] "
        f"{ids[0]} → {ids[-1]}"
    )


    with torch.inference_mode():

        pred_chunk = (

            pipeline

            .predict_df(

                ctx_chunk,

                future_df=
                    fut_chunk,

                id_column=
                    "item_id",

                timestamp_column=
                    "timestamp",

                target=
                    "target",

                prediction_length=
                    PRED_LEN,

                quantile_levels=[
                    0.5
                ],

                context_length=
                    CONTEXT_LEN,

                batch_size=
                    BATCH_SIZE,

                cross_learning=
                    False,

                freq=
                    "15min",
            )
        )


    all_pred_parts.append(
        pred_chunk
    )


    print(
        "  points =",
        len(pred_chunk)
    )


    del ctx_chunk
    del fut_chunk
    del pred_chunk

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 12. 整理预测
# ============================================================

raw_pred = pd.concat(

    all_pred_parts,

    ignore_index=True
)


assert (
    len(raw_pred)
    ==
    507 * 96
)


pred = raw_pred[
    [
        "item_id",
        "timestamp",
        "predictions",
    ]
].copy()


pred[
    "timestamp"
] = pd.to_datetime(

    pred[
        "timestamp"
    ]
)


pred[
    "target_date"
] = pd.to_datetime(

    pred[
        "item_id"
    ]
)


pred = pred.rename(

    columns={

        "predictions":
            "pred_chronos"
    }
)


# ============================================================
# 13. 合并 truth
# ============================================================

result = pred.merge(

    truth,

    on=[
        "target_date",
        "timestamp",
    ],

    how="left",

    validate="one_to_one",
)


assert (
    len(result)
    ==
    507 * 96
)


# ============================================================
# 14. Fold / month
# ============================================================

result[
    "fold"
] = np.select(

    [

        (
            result["target_date"]
            >=
            pd.Timestamp(
                "2025-04-01"
            )
        )

        &

        (
            result["target_date"]
            <
            pd.Timestamp(
                "2025-07-01"
            )
        ),

        (
            result["target_date"]
            >=
            pd.Timestamp(
                "2025-07-01"
            )
        )

        &

        (
            result["target_date"]
            <
            pd.Timestamp(
                "2025-10-01"
            )
        ),

        (
            result["target_date"]
            >=
            pd.Timestamp(
                "2025-10-01"
            )
        )

        &

        (
            result["target_date"]
            <
            pd.Timestamp(
                "2026-01-01"
            )
        ),
    ],

    [
        1,
        2,
        3,
    ],

    default=0,
)


result[
    "Month"
] = (

    result[
        "target_date"
    ]

    .dt
    .strftime(
        "%Y-%m"
    )
)


# ============================================================
# 15. 合法评价点
# ============================================================

score = result[

    result[
        "load_valid"
    ]
    ==
    1

].copy()


assert (
    len(score)
    ==
    48247
)


# ============================================================
# 16. 2025 3-Fold
# ============================================================

cv_rows = []


for fold in [
    1,
    2,
    3
]:

    d = score[
        score["fold"]
        ==
        fold
    ]


    m = calc_metrics(

        d[
            "load_raw"
        ],

        d[
            "pred_chronos"
        ],
    )


    cv_rows.append({

        "Fold":
            f"Fold{fold}",

        "Points":
            len(d),

        **m,
    })


cv_summary = pd.DataFrame(
    cv_rows
)


assert (
    cv_summary[
        "Points"
    ].tolist()
    ==
    [
        8372,
        8832,
        8772,
    ]
)


cv_mean = float(

    cv_summary[
        "MAPE"
    ].mean()
)


cv_score = score[

    score[
        "fold"
    ].isin(
        [
            1,
            2,
            3
        ]
    )

]


assert (
    len(cv_score)
    ==
    25976
)


cv_pooled = calc_metrics(

    cv_score[
        "load_raw"
    ],

    cv_score[
        "pred_chronos"
    ],
)


# ============================================================
# 17. 2026 Jan-Aug
# ============================================================

score_2026 = score[

    score[
        "target_date"
    ]
    >=
    pd.Timestamp(
        "2026-01-01"
    )

].copy()


assert (
    len(score_2026)
    ==
    22271
)


month_rows = []


for month, d in (

    score_2026

    .groupby(
        "Month",
        sort=True
    )
):

    m = calc_metrics(

        d[
            "load_raw"
        ],

        d[
            "pred_chronos"
        ],
    )


    month_rows.append({

        "Month":
            month,

        "Points":
            len(d),

        **m,
    })


month_summary = pd.DataFrame(
    month_rows
)


monthly_mean = float(

    month_summary[
        "MAPE"
    ].mean()
)


pooled_2026 = calc_metrics(

    score_2026[
        "load_raw"
    ],

    score_2026[
        "pred_chronos"
    ],
)


# ============================================================
# 18. 保存新 Release 结果
# ============================================================

result.to_csv(

    NEW_PRED_PATH,

    index=False,
)


cv_summary.to_csv(

    OUT_DIR
    / "chronos_zero_shot_2025_cv.csv",

    index=False,
)


month_summary.to_csv(

    OUT_DIR
    / "chronos_zero_shot_2026_monthly.csv",

    index=False,
)


# ============================================================
# 19. 与原正式预测逐点比较（如果旧文件还在）
# ============================================================

max_pred_diff = None


if OLD_PRED_PATH.exists():

    old = pd.read_csv(
        OLD_PRED_PATH
    )


    old[
        "timestamp"
    ] = pd.to_datetime(
        old["timestamp"]
    )


    old[
        "target_date"
    ] = pd.to_datetime(
        old["target_date"]
    )


    old = old[
        [
            "target_date",
            "timestamp",
            "pred_chronos",
        ]
    ].rename(

        columns={
            "pred_chronos":
                "pred_old"
        }
    )


    check = result.merge(

        old,

        on=[
            "target_date",
            "timestamp"
        ],

        how="inner",

        validate="one_to_one",
    )


    assert (
        len(check)
        ==
        507 * 96
    )


    max_pred_diff = float(

        np.max(

            np.abs(

                check[
                    "pred_chronos"
                ]

                -

                check[
                    "pred_old"
                ]
            )
        )
    )


# ============================================================
# 20. FINAL CHECK
# ============================================================

print("\n")
print("=" * 110)
print("CHRONOS ZERO-SHOT RELEASE REPRODUCTION FINAL CHECK")
print("=" * 110)


print("\n2025:")


for _, row in cv_summary.iterrows():

    ref = REF_2025[
        row["Fold"]
    ]


    print(

        f"{row['Fold']:5s}"

        f" ref={ref:.6f}%"

        f" | new={row['MAPE']:.6f}%"

        f" | diff="
        f"{row['MAPE']-ref:+.6f} pp"
    )


print(

    f"\n3-Fold Mean:"

    f" ref={REF_2025_MEAN:.6f}%"

    f" | new={cv_mean:.6f}%"

    f" | diff="
    f"{cv_mean-REF_2025_MEAN:+.6f} pp"
)


print(

    f"Pooled MAPE:"

    f" ref={REF_2025_POOLED:.6f}%"

    f" | new={cv_pooled['MAPE']:.6f}%"
)


print("\n2026 monthly:")


print(

    month_summary[
        [
            "Month",
            "Points",
            "MAPE",
            "WMAPE",
        ]
    ]

    .to_string(

        index=False,

        float_format=
            lambda x:
                f"{x:.6f}"
    )
)


print(

    f"\n2026 Monthly Mean:"

    f" ref={REF_2026_MONTHLY_MEAN:.6f}%"

    f" | new={monthly_mean:.6f}%"

    f" | diff="
    f"{monthly_mean-REF_2026_MONTHLY_MEAN:+.6f} pp"
)


print(

    f"2026 Pooled MAPE:"

    f" ref={REF_2026_POOLED:.6f}%"

    f" | new={pooled_2026['MAPE']:.6f}%"
)


print(

    f"2026 scored points:"

    f" ref={REF_2026_POINTS}"

    f" | new={len(score_2026)}"
)


if max_pred_diff is not None:

    print(

        f"\nMax absolute prediction difference "
        f"vs original saved run = "
        f"{max_pred_diff:.10f}"
    )


# ============================================================
# 21. 保存 summary
# ============================================================

summary = {

    "model":
        "Chronos-2 Zero-shot Final",

    "model_id":
        MODEL_ID,

    "context_days":
        CONTEXT_DAYS,

    "context_length":
        CONTEXT_LEN,

    "prediction_length":
        PRED_LEN,

    "weather_cols":
        WEATHER_COLS,

    "cross_learning":
        False,

    "2025_fold_mape": {

        row["Fold"]:
            float(
                row["MAPE"]
            )

        for _, row
        in cv_summary.iterrows()
    },

    "2025_mean_mape":
        cv_mean,

    "2025_pooled_mape":
        cv_pooled[
            "MAPE"
        ],

    "2026_monthly_mean_mape":
        monthly_mean,

    "2026_pooled_mape":
        pooled_2026[
            "MAPE"
        ],

    "2026_scored_points":
        len(score_2026),

    "max_prediction_diff_vs_old":
        max_pred_diff,
}


with open(

    OUT_DIR
    / "reproduction_summary.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        summary,

        f,

        ensure_ascii=False,

        indent=2
    )


print(
    "\nSaved to:"
)

print(
    OUT_DIR
)

print(
    "\n✅ CHRONOS ZERO-SHOT RELEASE REPRODUCTION FINISHED"
)