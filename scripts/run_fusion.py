# ============================================================
# GitHub Release Reproduction
# FINAL STEP — Ridge Fusion V2
#
# Models:
#   LightGBM
#   TFT
#   Chronos Full FT
#
# 正式方法：
#   只用2025选择 Ridge alpha
#
#   F1      -> F2
#   F1 + F2 -> F3
#   两个未来验证 MAPE 平均 -> alpha
#
#   alpha冻结
#   全部2025重新拟合 Ridge
#   -> 2026 untouched test
#
# 同时做：
#   LGB + TFT
#   LGB + Chronos
#   TFT + Chronos
#   LGB + TFT + Chronos
#
# pair结果仅作消融，不用于选择最终模型
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# 0. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# LGB + TFT 已经严格逐点对齐的底板
OLD_FUSION_DIR = (
    ROOT
    / "data" / "model_inputs" / "fusion"
)

BASE_2025_PATH = (
    OLD_FUSION_DIR
    / "aligned_2025_three_models_and_fusions.csv.gz"
)

BASE_2026_PATH = (
    OLD_FUSION_DIR
    / "aligned_2026_three_models_and_fusions.csv.gz"
)


# 本次 Release Full FT
FULLFT_2025_PATH = (
    ROOT
    / "results" / "reproduction"
    / "chronos_fullft"
    / "2025"
    / "chronos_fullft_release_2025_predictions.csv"
)

FULLFT_2026_PATH = (
    ROOT
    / "results" / "reproduction"
    / "chronos_fullft"
    / "2026_rolling"
    / "chronos_fullft_release_2026_predictions.csv"
)


OUT_DIR = (
    ROOT
    / "results" / "reproduction"
    / "fusion_final_ridge"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 115)
print("FINAL RIDGE FUSION — RELEASE REPRODUCTION")
print("=" * 115)

for p in [
    BASE_2025_PATH,
    BASE_2026_PATH,
    FULLFT_2025_PATH,
    FULLFT_2026_PATH,
]:
    print(
        "✅" if p.exists() else "❌",
        p
    )

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# 1. READ LGB + TFT BASE
# ============================================================

base25 = pd.read_csv(
    BASE_2025_PATH
)

base26 = pd.read_csv(
    BASE_2026_PATH
)


for df in [
    base25,
    base26,
]:
    df["target_date"] = (
        pd.to_datetime(
            df["target_date"]
        )
        .dt.normalize()
    )

    df["slot"] = pd.to_numeric(
        df["slot"]
    ).astype(int)


BASE_COLS = [
    "target_date",
    "slot",
    "actual",
    "pred_lgb",
    "pred_tft",
]


for name, df in [
    ("2025", base25),
    ("2026", base26),
]:
    missing = [
        c
        for c in BASE_COLS
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name} base missing: {missing}"
        )


base25 = base25[
    BASE_COLS
].copy()

base26 = base26[
    BASE_COLS
].copy()


# ============================================================
# 2. READ RELEASE FULL FT 2025
# ============================================================

full25 = pd.read_csv(
    FULLFT_2025_PATH
)

full25["target_date"] = (
    pd.to_datetime(
        full25["target_date"]
    )
    .dt.normalize()
)


if "slot" in full25.columns:
    full25["slot"] = pd.to_numeric(
        full25["slot"]
    ).astype(int)

elif "point" in full25.columns:
    full25["slot"] = pd.to_numeric(
        full25["point"]
    ).astype(int)

else:
    raise RuntimeError(
        "2025 Full FT 找不到 slot / point"
    )


full25 = (
    full25[
        [
            "target_date",
            "slot",
            "y_true",
            "y_pred",
        ]
    ]
    .rename(
        columns={
            "y_true":
                "actual_fullft",

            "y_pred":
                "pred_chronos_fullft",
        }
    )
)


# ============================================================
# 3. READ RELEASE FULL FT 2026
# ============================================================

full26 = pd.read_csv(
    FULLFT_2026_PATH
)

full26["target_date"] = (
    pd.to_datetime(
        full26["target_date"]
    )
    .dt.normalize()
)

full26["slot"] = pd.to_numeric(
    full26["slot"]
).astype(int)


full26 = (
    full26[
        [
            "target_date",
            "slot",
            "load_actual",
            "pred_chronos_fullft_final_rolling",
        ]
    ]
    .rename(
        columns={
            "load_actual":
                "actual_fullft",

            "pred_chronos_fullft_final_rolling":
                "pred_chronos_fullft",
        }
    )
)


# ============================================================
# 4. DUPLICATE CHECK
# ============================================================

KEYS = [
    "target_date",
    "slot",
]


for name, df in [
    ("base25", base25),
    ("base26", base26),
    ("full25", full25),
    ("full26", full26),
]:
    dup = df.duplicated(
        KEYS,
        keep=False
    )

    if dup.any():
        raise RuntimeError(
            f"{name} duplicate keys = {dup.sum()}"
        )


# ============================================================
# 5. STRICT ALIGNMENT
# ============================================================

df25 = base25.merge(
    full25,
    on=KEYS,
    how="inner",
    validate="one_to_one",
)

df26 = base26.merge(
    full26,
    on=KEYS,
    how="inner",
    validate="one_to_one",
)


truth_diff25 = np.nanmax(
    np.abs(
        df25["actual"].to_numpy(float)
        -
        df25["actual_fullft"].to_numpy(float)
    )
)

truth_diff26 = np.nanmax(
    np.abs(
        df26["actual"].to_numpy(float)
        -
        df26["actual_fullft"].to_numpy(float)
    )
)


# 正式有效点
def valid_rows(df):

    return (
        np.isfinite(
            df["actual"].to_numpy(float)
        )
        &
        (
            df["actual"].to_numpy(float)
            >
            0
        )
        &
        np.isfinite(
            df["pred_lgb"].to_numpy(float)
        )
        &
        np.isfinite(
            df["pred_tft"].to_numpy(float)
        )
        &
        np.isfinite(
            df[
                "pred_chronos_fullft"
            ].to_numpy(float)
        )
    )


df25 = (
    df25.loc[
        valid_rows(df25)
    ]
    .copy()
)

df26 = (
    df26.loc[
        valid_rows(df26)
    ]
    .copy()
)


assert len(df25) == 25976, len(df25)
assert len(df26) == 22271, len(df26)


if truth_diff25 > 1e-4:
    raise RuntimeError(
        f"2025 truth mismatch: {truth_diff25}"
    )

if truth_diff26 > 1e-4:
    raise RuntimeError(
        f"2026 truth mismatch: {truth_diff26}"
    )


df25.drop(
    columns="actual_fullft",
    inplace=True
)

df26.drop(
    columns="actual_fullft",
    inplace=True
)


print("\n" + "=" * 115)
print("STRICT ALIGNMENT CHECK")
print("=" * 115)

print(
    "2025 points =",
    len(df25)
)

print(
    "2026 points =",
    len(df26)
)

print(
    "2025 truth max diff =",
    truth_diff25
)

print(
    "2026 truth max diff =",
    truth_diff26
)

print(
    "✅ Strict alignment passed"
)


# ============================================================
# 6. FOLD / MONTH
# ============================================================

df25["fold"] = np.select(

    [
        (
            df25["target_date"]
            <
            pd.Timestamp(
                "2025-07-01"
            )
        ),

        (
            (
                df25["target_date"]
                >=
                pd.Timestamp(
                    "2025-07-01"
                )
            )
            &
            (
                df25["target_date"]
                <
                pd.Timestamp(
                    "2025-10-01"
                )
            )
        ),

        (
            df25["target_date"]
            >=
            pd.Timestamp(
                "2025-10-01"
            )
        ),
    ],

    [
        "Fold1",
        "Fold2",
        "Fold3",
    ],

    default="UNKNOWN",
)


if (
    df25["fold"]
    ==
    "UNKNOWN"
).any():

    raise RuntimeError(
        "存在未知 Fold"
    )


df26["month"] = (
    df26[
        "target_date"
    ]
    .dt.to_period("M")
    .astype(str)
)


print("\n2025 Fold points:")
print(
    df25["fold"]
    .value_counts()
    .sort_index()
)

print("\n2026 Month points:")
print(
    df26["month"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. METRICS
# ============================================================

def calc_mape(
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

    return float(
        np.mean(
            np.abs(
                p - y
            )
            /
            y
        )
        *
        100
    )


def calc_all(
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
# 8. RIDGE
#
# X 标准化
# intercept 不正则
# y 保持 MW
# ============================================================

def fit_ridge(
    X,
    y,
    alpha
):

    X = np.asarray(
        X,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )


    x_mean = X.mean(
        axis=0
    )

    x_std = X.std(
        axis=0,
        ddof=0
    )

    x_std[
        x_std < 1e-12
    ] = 1.0


    Z = (
        X - x_mean
    ) / x_std


    intercept = float(
        y.mean()
    )

    yc = (
        y - intercept
    )


    A = (
        Z.T @ Z
        +
        float(alpha)
        *
        np.eye(
            Z.shape[1]
        )
    )

    b = (
        Z.T @ yc
    )


    coef = np.linalg.solve(
        A,
        b
    )


    return {
        "alpha":
            float(alpha),

        "x_mean":
            x_mean,

        "x_std":
            x_std,

        "intercept":
            intercept,

        "coef":
            coef,
    }


def predict_ridge(
    model,
    X
):

    X = np.asarray(
        X,
        dtype=float
    )

    Z = (
        X
        -
        model["x_mean"]
    ) / model["x_std"]


    return (
        model["intercept"]
        +
        Z
        @
        model["coef"]
    )


# ============================================================
# 9. ALPHA GRID
# ============================================================

ALPHAS = [
    0.0,
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]


COMBINATIONS = {

    "LGB+TFT":
        [
            "pred_lgb",
            "pred_tft",
        ],

    "LGB+Chronos":
        [
            "pred_lgb",
            "pred_chronos_fullft",
        ],

    "TFT+Chronos":
        [
            "pred_tft",
            "pred_chronos_fullft",
        ],

    "LGB+TFT+Chronos":
        [
            "pred_lgb",
            "pred_tft",
            "pred_chronos_fullft",
        ],
}


f1 = df25[
    df25["fold"]
    ==
    "Fold1"
].copy()

f2 = df25[
    df25["fold"]
    ==
    "Fold2"
].copy()

f3 = df25[
    df25["fold"]
    ==
    "Fold3"
].copy()

f12 = df25[
    df25["fold"]
    .isin(
        [
            "Fold1",
            "Fold2",
        ]
    )
].copy()


# ============================================================
# 10. EACH COMBINATION:
#
# alpha:
#   F1 -> F2
#   F1+F2 -> F3
#
# then all 2025 fit final model
# ============================================================

alpha_rows = []
parameter_rows = []
ablation_rows = []

all3_model = None


for combo_name, features in (
    COMBINATIONS.items()
):

    search_rows = []


    for alpha in ALPHAS:

        # -------------------------
        # F1 -> F2
        # -------------------------

        m1 = fit_ridge(

            f1[
                features
            ].to_numpy(),

            f1[
                "actual"
            ].to_numpy(),

            alpha,
        )


        p2 = predict_ridge(

            m1,

            f2[
                features
            ].to_numpy(),
        )


        mape2 = calc_mape(

            f2[
                "actual"
            ],

            p2,
        )


        # -------------------------
        # F1+F2 -> F3
        # -------------------------

        m12 = fit_ridge(

            f12[
                features
            ].to_numpy(),

            f12[
                "actual"
            ].to_numpy(),

            alpha,
        )


        p3 = predict_ridge(

            m12,

            f3[
                features
            ].to_numpy(),
        )


        mape3 = calc_mape(

            f3[
                "actual"
            ],

            p3,
        )


        chrono_mean = (
            mape2
            +
            mape3
        ) / 2.0


        row = {
            "Combination":
                combo_name,

            "alpha":
                alpha,

            "F1_to_F2_MAPE":
                mape2,

            "F12_to_F3_MAPE":
                mape3,

            "Chronological_Mean_MAPE":
                chrono_mean,
        }


        alpha_rows.append(
            row
        )

        search_rows.append(
            row
        )


    search_df = (
        pd.DataFrame(
            search_rows
        )
        .sort_values(
            [
                "Chronological_Mean_MAPE",
                "alpha",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    best_alpha = float(
        search_df.iloc[0][
            "alpha"
        ]
    )


    # -----------------------------
    # Final fit: all 2025
    # -----------------------------

    final_model = fit_ridge(

        df25[
            features
        ].to_numpy(),

        df25[
            "actual"
        ].to_numpy(),

        best_alpha,
    )


    # 2026 untouched prediction
    pred26 = predict_ridge(

        final_model,

        df26[
            features
        ].to_numpy(),
    )


    temp = df26[
        [
            "target_date",
            "slot",
            "month",
            "actual",
        ]
    ].copy()

    temp["pred"] = pred26


    monthly_mapes = (

        temp

        .groupby(
            "month",
            sort=True
        )

        .apply(
            lambda g:
                calc_mape(
                    g["actual"],
                    g["pred"]
                ),
            include_groups=False
        )
    )


    overall = calc_all(

        temp[
            "actual"
        ],

        temp[
            "pred"
        ],
    )


    monthly_mean = float(
        monthly_mapes.mean()
    )


    ablation_rows.append({

        "Combination":
            combo_name,

        "Num_Models":
            len(features),

        "Best_Alpha":
            best_alpha,

        "Chronological_2025_MAPE":
            float(
                search_df.iloc[0][
                    "Chronological_Mean_MAPE"
                ]
            ),

        "2026_Monthly_Mean_MAPE":
            monthly_mean,

        "2026_Pooled_MAPE":
            overall["MAPE"],

        "2026_Pooled_WMAPE":
            overall["WMAPE"],

        "2026_MAE":
            overall["MAE"],

        "2026_RMSE":
            overall["RMSE"],
    })


    for i, feature in enumerate(
        features
    ):

        parameter_rows.append({

            "Combination":
                combo_name,

            "Parameter":
                feature,

            "Value":
                float(
                    final_model[
                        "coef"
                    ][i]
                ),
        })


    parameter_rows.append({

        "Combination":
            combo_name,

        "Parameter":
            "intercept",

        "Value":
            float(
                final_model[
                    "intercept"
                ]
            ),
    })


    parameter_rows.append({

        "Combination":
            combo_name,

        "Parameter":
            "alpha",

        "Value":
            best_alpha,
    })


    if (
        combo_name
        ==
        "LGB+TFT+Chronos"
    ):
        all3_model = final_model


# ============================================================
# 11. TABLES
# ============================================================

alpha_search = pd.DataFrame(
    alpha_rows
)

params = pd.DataFrame(
    parameter_rows
)

ablation = (
    pd.DataFrame(
        ablation_rows
    )
    .sort_values(
        [
            "Num_Models",
            "Combination",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 12. FINAL ALL-3 PREDICTIONS
# ============================================================

ALL3_FEATURES = (
    COMBINATIONS[
        "LGB+TFT+Chronos"
    ]
)


df25_out = df25.copy()

df25_out[
    "pred_ridge_final"
] = predict_ridge(

    all3_model,

    df25[
        ALL3_FEATURES
    ].to_numpy(),
)


df26_out = df26.copy()

df26_out[
    "pred_ridge_final"
] = predict_ridge(

    all3_model,

    df26[
        ALL3_FEATURES
    ].to_numpy(),
)


# ============================================================
# 13. MONTHLY ALL-3
# ============================================================

monthly_rows = []


for month, g in (
    df26_out.groupby(
        "month",
        sort=True
    )
):

    m = calc_all(

        g["actual"],

        g[
            "pred_ridge_final"
        ],
    )


    monthly_rows.append({

        "Month":
            month,

        "Scored_Points":
            len(g),

        **m,
    })


monthly_all3 = pd.DataFrame(
    monthly_rows
)


overall_all3 = calc_all(

    df26_out[
        "actual"
    ],

    df26_out[
        "pred_ridge_final"
    ],
)


monthly_mean_all3 = float(
    monthly_all3[
        "MAPE"
    ].mean()
)


# ============================================================
# 14. HISTORICAL REFERENCE
#
# Full FT 本次略有GPU漂移，
# 所以含Chronos的融合不要求逐位完全一致。
#
# LGB+TFT应基本完全一致。
# ============================================================

REFERENCE = {

    "LGB+TFT":
        2.224383,

    "LGB+Chronos":
        1.924466,

    "TFT+Chronos":
        1.916504,

    "LGB+TFT+Chronos":
        1.905930,
}


# ============================================================
# 15. FINAL PRINT
# ============================================================

print("\n" + "=" * 125)
print("RIDGE ALPHA — 2025 STRICT CHRONOLOGICAL VALIDATION")
print("=" * 125)


best_alpha_table = (

    alpha_search

    .sort_values(
        [
            "Combination",
            "Chronological_Mean_MAPE",
            "alpha",
        ]
    )

    .groupby(
        "Combination",
        as_index=False
    )

    .first()
)


print(
    best_alpha_table[
        [
            "Combination",
            "alpha",
            "F1_to_F2_MAPE",
            "F12_to_F3_MAPE",
            "Chronological_Mean_MAPE",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print("\n" + "=" * 125)
print("2026 RIDGE PAIR ABLATION")
print("=" * 125)


show = ablation.copy()

show[
    "Historical_Reference"
] = show[
    "Combination"
].map(
    REFERENCE
)

show[
    "Diff_vs_Historical"
] = (
    show[
        "2026_Monthly_Mean_MAPE"
    ]
    -
    show[
        "Historical_Reference"
    ]
)


print(
    show[
        [
            "Combination",
            "Best_Alpha",
            "2026_Monthly_Mean_MAPE",
            "Historical_Reference",
            "Diff_vs_Historical",
            "2026_Pooled_MAPE",
            "2026_Pooled_WMAPE",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print("\n" + "=" * 125)
print("FINAL ALL-3 RIDGE — 2026")
print("=" * 125)


print(
    monthly_all3.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print("\nOVERALL")

print(
    f"Monthly Mean MAPE = "
    f"{monthly_mean_all3:.6f}%"
)

print(
    f"Pooled MAPE       = "
    f"{overall_all3['MAPE']:.6f}%"
)

print(
    f"Pooled WMAPE      = "
    f"{overall_all3['WMAPE']:.6f}%"
)

print(
    f"MAE               = "
    f"{overall_all3['MAE']:.6f} MW"
)

print(
    f"RMSE              = "
    f"{overall_all3['RMSE']:.6f} MW"
)

print(
    f"Scored points     = "
    f"{len(df26_out)}"
)

print(
    "\nHistorical Final Ridge "
    "= 1.905930% monthly mean"
)

print(
    "Release difference      = "
    f"{monthly_mean_all3 - 1.905930:+.6f} pp"
)


# ============================================================
# 16. SAVE
# ============================================================

alpha_search.to_csv(

    OUT_DIR
    / "ridge_alpha_search_all_combinations.csv",

    index=False
)


params.to_csv(

    OUT_DIR
    / "ridge_final_parameters.csv",

    index=False
)


ablation.to_csv(

    OUT_DIR
    / "leave_one_model_out_ablation.csv",

    index=False
)


best_alpha_table.to_csv(

    OUT_DIR
    / "ridge_fold3_validation_ablation.csv",

    index=False
)


monthly_all3.to_csv(

    OUT_DIR
    / "ridge_2026_monthly_mape.csv",

    index=False
)


pd.DataFrame(
    [
        {
            "Model":
                "Ridge Final",

            "Monthly_Mean_MAPE":
                monthly_mean_all3,

            "Pooled_MAPE":
                overall_all3[
                    "MAPE"
                ],

            "Pooled_WMAPE":
                overall_all3[
                    "WMAPE"
                ],

            "MAE_MW":
                overall_all3[
                    "MAE"
                ],

            "RMSE_MW":
                overall_all3[
                    "RMSE"
                ],

            "Scored_Points":
                len(df26_out),
        }
    ]
).to_csv(

    OUT_DIR
    / "ridge_2026_summary.csv",

    index=False
)


df25_out.to_csv(

    OUT_DIR
    / "ridge_predictions_2025.csv",

    index=False
)


df26_out.to_csv(

    OUT_DIR
    / "ridge_predictions_2026.csv",

    index=False
)


config = {

    "final_fusion_method":
        "Ridge Stacking",

    "method_selection":
        "2025 chronological validation only",

    "alpha_protocol":
        "F1->F2 and F1+F2->F3 mean MAPE",

    "final_fit":
        "all 2025 valid aligned points",

    "final_test":
        "2026 Jan-Aug untouched",

    "models": [
        "LightGBM V14",
        "TFT V22-V1",
        "Chronos-2 Full Fine-tuning",
    ],

    "historical_final_monthly_mean_mape":
        1.905930,

    "release_monthly_mean_mape":
        monthly_mean_all3,

    "release_pooled_mape":
        overall_all3["MAPE"],

    "release_pooled_wmape":
        overall_all3["WMAPE"],

    "points_2025":
        len(df25),

    "points_2026":
        len(df26),
}


with open(

    OUT_DIR
    / "final_v2_config.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nSaved to:")
print(OUT_DIR)

print(
    "\n✅ FINAL RIDGE RELEASE REPRODUCTION FINISHED"
)