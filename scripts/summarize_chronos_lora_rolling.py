# ============================================================
# Chronos LoRA 2026 Release
# 修复汇总阶段，不重新训练
#
# 1. 检查 Jan-Aug 结果是否全部保存
# 2. 检查 LoRA checkpoint 是否存在
# 3. 使用真实预测列重新计算月度/总体指标
# 4. 保存最终 Release 汇总
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    ROOT
    / "results" / "reproduction"
    / "chronos_lora"
    / "2026_rolling"
)

MONTHS = [
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
    "2026-07",
    "2026-08",
]

REFERENCE = {
    "2026-01": 1.585980,
    "2026-02": 2.695482,
    "2026-03": 1.890535,
    "2026-04": 2.083609,
    "2026-05": 2.330953,
    "2026-06": 1.897937,
    "2026-07": 1.864168,
    "2026-08": 1.576153,
}

REF_MEAN = 1.990602
REF_POOLED = 2.001122
REF_WMAPE = 1.940015
REF_POINTS = 22271


# ============================================================
# 1. 检查八个月保存状态
# ============================================================

print("=" * 110)
print("1. CHECK SAVED MONTHLY RESULTS")
print("=" * 110)

pred_parts = []
result_rows = []

all_ok = True


for month in MONTHS:

    tag = month.replace("-", "_")

    month_dir = (
        OUT_DIR
        / tag
    )

    result_path = (
        month_dir
        / "month_result.json"
    )

    pred_path = (
        month_dir
        / "predictions.csv"
    )

    # 查 checkpoint / adapter
    checkpoint_hits = list(
        month_dir.rglob(
            "*final_lora_step1400*"
        )
    )

    adapter_hits = list(
        month_dir.rglob(
            "adapter_config.json"
        )
    )


    result_ok = result_path.exists()
    pred_ok = pred_path.exists()

    print(
        f"{month}"
        f" | result={'✅' if result_ok else '❌'}"
        f" | prediction={'✅' if pred_ok else '❌'}"
        f" | checkpoint_hits={len(checkpoint_hits)}"
        f" | adapter_hits={len(adapter_hits)}"
    )


    if not (
        result_ok
        and pred_ok
    ):
        all_ok = False
        continue


    with open(
        result_path,
        "r",
        encoding="utf-8"
    ) as f:

        r = json.load(f)

    r["month"] = month

    result_rows.append(r)


    temp = pd.read_csv(
        pred_path
    )

    temp["month"] = month

    pred_parts.append(temp)


if not all_ok:

    raise RuntimeError(
        "至少有一个月份缺少 result/prediction，"
        "先不要继续汇总。"
    )


print(
    "\n✅ Jan-Aug 八个月训练结果全部存在"
)


# ============================================================
# 2. 合并八个月预测
# ============================================================

all_pred = pd.concat(
    pred_parts,
    ignore_index=True
)


print("\n" + "=" * 110)
print("2. PREDICTION DATA")
print("=" * 110)

print(
    "shape =",
    all_pred.shape
)

print(
    "columns =",
    all_pred.columns.tolist()
)


# ============================================================
# 3. 正确预测列
# ============================================================

PRED_COL = (
    "pred_chronos_lora_v3_nogap_rolling"
)

ACTUAL_COL = "load_actual"
VALID_COL = "load_valid"


for c in [
    PRED_COL,
    ACTUAL_COL,
    VALID_COL,
]:

    if c not in all_pred.columns:

        raise KeyError(
            f"缺少列：{c}"
        )


print(
    "\nPrediction column =",
    PRED_COL
)


# ============================================================
# 4. 正式评分 mask
# ============================================================

score = all_pred[

    all_pred[
        VALID_COL
    ].eq(1)

    &

    all_pred[
        ACTUAL_COL
    ].notna()

    &

    all_pred[
        PRED_COL
    ].notna()

    &

    (
        all_pred[
            ACTUAL_COL
        ]
        >
        0
    )

].copy()


print(
    "Scored points =",
    len(score)
)


assert (
    len(score)
    ==
    REF_POINTS
), (
    f"评分点数异常："
    f"{len(score)} != {REF_POINTS}"
)


# ============================================================
# 5. Metrics
# ============================================================

def metrics(df):

    y = (
        df[
            ACTUAL_COL
        ]
        .to_numpy(
            dtype=float
        )
    )

    p = (
        df[
            PRED_COL
        ]
        .to_numpy(
            dtype=float
        )
    )

    ae = np.abs(
        p - y
    )

    return {

        "MAPE":
            float(
                np.mean(
                    ae / y
                )
                * 100
            ),

        "WMAPE":
            float(
                ae.sum()
                /
                y.sum()
                * 100
            ),

        "MAE":
            float(
                ae.mean()
            ),

        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        (p-y) ** 2
                    )
                )
            ),
    }


# ============================================================
# 6. 月度指标
# ============================================================

rows = []


for month in MONTHS:

    d = score[
        score["month"]
        ==
        month
    ]

    m = metrics(d)

    rows.append({

        "Month":
            month,

        "Scored_Points":
            len(d),

        **m,
    })


monthly = pd.DataFrame(
    rows
)


monthly_mean = float(
    monthly[
        "MAPE"
    ].mean()
)


pooled = metrics(
    score
)


# ============================================================
# 7. FINAL CHECK
# ============================================================

print("\n" + "=" * 115)
print(
    "CHRONOS LoRA — 2026 RELEASE REPRODUCTION FINAL CHECK"
)
print("=" * 115)


for _, row in monthly.iterrows():

    month = row["Month"]

    ref = REFERENCE[
        month
    ]

    print(

        f"{month}"

        f" | points={int(row['Scored_Points'])}"

        f" | ref={ref:.6f}%"

        f" | new={row['MAPE']:.6f}%"

        f" | diff="
        f"{row['MAPE']-ref:+.6f} pp"
    )


print("\nOVERALL")


print(
    f"Monthly Mean MAPE:"
    f" ref={REF_MEAN:.6f}%"
    f" | new={monthly_mean:.6f}%"
    f" | diff={monthly_mean-REF_MEAN:+.6f} pp"
)


print(
    f"Pooled MAPE:"
    f" ref={REF_POOLED:.6f}%"
    f" | new={pooled['MAPE']:.6f}%"
    f" | diff={pooled['MAPE']-REF_POOLED:+.6f} pp"
)


print(
    f"Pooled WMAPE:"
    f" ref={REF_WMAPE:.6f}%"
    f" | new={pooled['WMAPE']:.6f}%"
    f" | diff={pooled['WMAPE']-REF_WMAPE:+.6f} pp"
)


print(
    f"Scored points:"
    f" ref={REF_POINTS}"
    f" | new={len(score)}"
)


# ============================================================
# 8. 保存最终 Release 汇总
# ============================================================

monthly.to_csv(

    OUT_DIR
    / "chronos_lora_release_2026_monthly.csv",

    index=False
)


all_pred.to_csv(

    OUT_DIR
    / "chronos_lora_release_2026_predictions.csv",

    index=False
)


with open(

    OUT_DIR
    / "reproduction_summary.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        {
            "model":
                "Chronos-2 LoRA Final",

            "fixed_stop_step":
                1400,

            "batch_size":
                28,

            "monthly_mean_mape":
                monthly_mean,

            "pooled_mape":
                pooled["MAPE"],

            "pooled_wmape":
                pooled["WMAPE"],

            "pooled_mae":
                pooled["MAE"],

            "pooled_rmse":
                pooled["RMSE"],

            "scored_points":
                int(
                    len(score)
                ),

            "monthly_mape": {

                row["Month"]:
                    float(
                        row["MAPE"]
                    )

                for _, row
                in monthly.iterrows()
            },
        },

        f,

        ensure_ascii=False,

        indent=2,
    )


print("\n✅ 汇总完成，没有重新训练")
print("Saved to:")
print(OUT_DIR)