# ============================================================
# GitHub Release Reproduction
# Step 1 — LightGBM V14 Final
#
# 重新复现：
#   2025 NO-GAP 3-Fold CV
#   2026 Jan-Aug Expanding Monthly Rolling
#
# 不修改、不覆盖历史正式结果
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb


# ============================================================
# 0. 路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CV_DATA_PATH = (
    ROOT
    / "data" / "model_inputs" / "lightgbm" / "lightgbm_v14_valid_supervised_v1.pkl.gz"
)

MASTER_PATH = (
    ROOT
    / "data" / "model_inputs" / "lightgbm" / "lightgbm_v14_master_v1.pkl.gz"
)

FEATURE_TEMPLATE_PATH = (
    ROOT
    / "data" / "model_inputs" / "lightgbm" / "lightgbm_v14_feature_header.csv"
)

OUT_DIR = (
    ROOT
    / "results" / "reproduction"
    / "lightgbm"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "系统负荷实际值(MW)"
DATE_COL = "目标日期"
TIME_COL = "timestamp"


# ============================================================
# 1. 正式冻结参数
# ============================================================

BASE_PARAMS = {

    "objective":
        "regression_l1",

    "learning_rate":
        0.0176184,

    "num_leaves":
        30,

    "max_depth":
        11,

    "min_child_samples":
        110,

    "colsample_bytree":
        0.660857,

    "subsample":
        0.783429,

    "subsample_freq":
        1,

    "reg_alpha":
        0.013741,

    "reg_lambda":
        0.109744,

    "random_state":
        42,

    "n_jobs":
        24,

    "verbosity":
        -1,
}


# ============================================================
# 2. 读取最终 48 个特征
# ============================================================

template_cols = pd.read_csv(
    FEATURE_TEMPLATE_PATH,
    nrows=0
).columns.tolist()

FEATURES = [
    c
    for c in template_cols
    if c not in [
        DATE_COL,
        TIME_COL,
        TARGET,
    ]
]

print("=" * 100)
print("LIGHTGBM RELEASE REPRODUCTION")
print("=" * 100)

print("Feature count =", len(FEATURES))

if len(FEATURES) != 48:
    raise RuntimeError(
        f"特征数量异常：{len(FEATURES)}，正式版本应为48"
    )

print("✅ 48 features confirmed")


# ============================================================
# 3. 指标函数
# ============================================================

def calc_metrics(y, pred):

    y = np.asarray(
        y,
        dtype=float
    )

    pred = np.asarray(
        pred,
        dtype=float
    )

    mae = np.mean(
        np.abs(y - pred)
    )

    rmse = np.sqrt(
        np.mean(
            (y - pred) ** 2
        )
    )

    mape = (
        np.mean(
            np.abs(
                (y - pred) / y
            )
        )
        * 100
    )

    wmape = (
        np.sum(
            np.abs(y - pred)
        )
        /
        np.sum(
            np.abs(y)
        )
        * 100
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "WMAPE": float(wmape),
    }


# ============================================================
# 4. 2025 NO-GAP 三折
# ============================================================

cv_data = pd.read_pickle(
    CV_DATA_PATH
).copy()

cv_data[DATE_COL] = pd.to_datetime(
    cv_data[DATE_COL]
).dt.normalize()

cv_data[TIME_COL] = pd.to_datetime(
    cv_data[TIME_COL]
)

cv_data = (
    cv_data
    .sort_values(TIME_COL)
    .reset_index(drop=True)
)

print("\nCV rows =", len(cv_data))


FOLDS = [

    {
        "fold": "Fold1",
        "train_end": "2025-03-31",
        "val_start": "2025-04-01",
        "val_end": "2025-06-30",
    },

    {
        "fold": "Fold2",
        "train_end": "2025-06-30",
        "val_start": "2025-07-01",
        "val_end": "2025-09-30",
    },

    {
        "fold": "Fold3",
        "train_end": "2025-09-30",
        "val_start": "2025-10-01",
        "val_end": "2025-12-31",
    },
]


cv_rows = []
cv_pred_parts = []
best_iterations = []


print("\n" + "=" * 100)
print("2025 3-FOLD REPRODUCTION")
print("=" * 100)


for fold_cfg in FOLDS:

    fold_name = fold_cfg["fold"]

    train_end = pd.Timestamp(
        fold_cfg["train_end"]
    )

    val_start = pd.Timestamp(
        fold_cfg["val_start"]
    )

    val_end = pd.Timestamp(
        fold_cfg["val_end"]
    )

    train = cv_data[
        cv_data[DATE_COL]
        <=
        train_end
    ].copy()

    val = cv_data[
        (
            cv_data[DATE_COL]
            >=
            val_start
        )
        &
        (
            cv_data[DATE_COL]
            <=
            val_end
        )
    ].copy()


    print(
        f"\n{fold_name}"
        f" | train={len(train)}"
        f" | val={len(val)}"
    )


    model = lgb.LGBMRegressor(
        **BASE_PARAMS,
        n_estimators=5000,
    )


    model.fit(

        train[FEATURES],
        train[TARGET],

        eval_set=[
            (
                val[FEATURES],
                val[TARGET],
            )
        ],

        eval_metric="l1",

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=False,
            ),

            lgb.log_evaluation(
                period=0
            ),
        ],
    )


    pred = model.predict(
        val[FEATURES],
        num_iteration=model.best_iteration_,
    )


    metric = calc_metrics(
        val[TARGET],
        pred
    )


    best_iterations.append(
        int(model.best_iteration_)
    )


    print(
        f"best_iteration = {model.best_iteration_}"
    )

    print(
        f"MAPE = {metric['MAPE']:.6f}%"
        f" | WMAPE = {metric['WMAPE']:.6f}%"
        f" | MAE = {metric['MAE']:.2f}"
        f" | RMSE = {metric['RMSE']:.2f}"
    )


    cv_rows.append({

        "Fold":
            fold_name,

        "Train_End":
            train_end,

        "Val_Start":
            val_start,

        "Val_End":
            val_end,

        "Train_Rows":
            len(train),

        "Val_Rows":
            len(val),

        "Best_Iteration":
            int(model.best_iteration_),

        **metric,
    })


    pred_part = val[
        [
            DATE_COL,
            TIME_COL,
            TARGET,
        ]
    ].copy()

    pred_part["Fold"] = fold_name
    pred_part["Prediction"] = pred

    cv_pred_parts.append(
        pred_part
    )


cv_result = pd.DataFrame(
    cv_rows
)

cv_predictions = pd.concat(
    cv_pred_parts,
    ignore_index=True
)


# ============================================================
# 5. 2025 汇总
# ============================================================

cv_mean_mape = float(
    cv_result["MAPE"].mean()
)

pooled_cv = calc_metrics(
    cv_predictions[TARGET],
    cv_predictions["Prediction"],
)

final_n_estimators = int(
    np.median(
        best_iterations
    )
)


print("\n" + "=" * 100)
print("2025 CV SUMMARY")
print("=" * 100)

print(
    cv_result[
        [
            "Fold",
            "Best_Iteration",
            "MAPE",
            "WMAPE",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print(
    f"\n3-Fold Mean MAPE = "
    f"{cv_mean_mape:.6f}%"
)

print(
    f"Pooled MAPE      = "
    f"{pooled_cv['MAPE']:.6f}%"
)

print(
    f"Median best_iteration = "
    f"{final_n_estimators}"
)


# ============================================================
# 6. 对照旧正式结果
# ============================================================

REF_FOLDS = {
    "Fold1": 2.417475,
    "Fold2": 2.569775,
    "Fold3": 2.079877,
}

REF_CV_MEAN = 2.355709
REF_N_ESTIMATORS = 1593


print("\nREFERENCE CHECK")

for fold_name, ref in REF_FOLDS.items():

    new = float(
        cv_result.loc[
            cv_result["Fold"] == fold_name,
            "MAPE"
        ].iloc[0]
    )

    print(
        f"{fold_name}: "
        f"old={ref:.6f}% | "
        f"new={new:.6f}% | "
        f"diff={new-ref:+.6f} pp"
    )


print(
    f"\n3Fold Mean:"
    f" old={REF_CV_MEAN:.6f}%"
    f" | new={cv_mean_mape:.6f}%"
    f" | diff={cv_mean_mape-REF_CV_MEAN:+.6f} pp"
)

print(
    f"Trees:"
    f" old={REF_N_ESTIMATORS}"
    f" | new={final_n_estimators}"
)


# ============================================================
# 7. 2026 Expanding Monthly Rolling
# ============================================================

master = pd.read_pickle(
    MASTER_PATH
).copy()

master[DATE_COL] = pd.to_datetime(
    master[DATE_COL]
).dt.normalize()

master[TIME_COL] = pd.to_datetime(
    master[TIME_COL]
)

master = (
    master
    .sort_values(TIME_COL)
    .reset_index(drop=True)
)


# Release 复现应使用 CV 决定的树数
# 同时正式值应该复现为1593
ROLLING_TREES = final_n_estimators


print("\n" + "=" * 100)
print("2026 JAN-AUG EXPANDING ROLLING")
print("=" * 100)

print(
    "Rolling trees =",
    ROLLING_TREES
)


monthly_rows = []
rolling_pred_parts = []


for month in range(1, 9):

    month_start = pd.Timestamp(
        year=2026,
        month=month,
        day=1
    )

    if month == 12:

        next_month = pd.Timestamp(
            "2027-01-01"
        )

    else:

        next_month = (
            month_start
            +
            pd.offsets.MonthBegin(1)
        )


    # ----------------------------------------
    # Train：预测月之前全部可评分历史
    # ----------------------------------------

    train_mask = (
        master[DATE_COL]
        <
        month_start
    )

    if "load_valid" in master.columns:

        train_mask &= (
            master["load_valid"]
            ==
            1
        )

    train_mask &= (
        master[TARGET].notna()
    )


    train = master[
        train_mask
    ].copy()


    # ----------------------------------------
    # Test：当月
    # ----------------------------------------

    test_mask = (
        (master[DATE_COL] >= month_start)
        &
        (master[DATE_COL] < next_month)
    )

    test = master[
        test_mask
    ].copy()


    if len(test) == 0:
        continue


    model = lgb.LGBMRegressor(
        **BASE_PARAMS,
        n_estimators=ROLLING_TREES,
    )


    model.fit(
        train[FEATURES],
        train[TARGET],
    )


    pred = model.predict(
        test[FEATURES]
    )


    # ----------------------------------------
    # 评价 mask
    # ----------------------------------------

    score_mask = (
        test[TARGET].notna()
    )

    if "load_valid" in test.columns:

        score_mask &= (
            test["load_valid"]
            ==
            1
        )


    metric = calc_metrics(

        test.loc[
            score_mask,
            TARGET
        ],

        pred[
            score_mask.to_numpy()
        ],
    )


    print(
        f"2026-{month:02d}"
        f" | train={len(train)}"
        f" | test={len(test)}"
        f" | scored={int(score_mask.sum())}"
        f" | MAPE={metric['MAPE']:.6f}%"
    )


    monthly_rows.append({

        "Month":
            month,

        "Train_Rows":
            len(train),

        "Test_Rows":
            len(test),

        "Scored_Rows":
            int(
                score_mask.sum()
            ),

        **metric,
    })


    pred_part = test[
        [
            DATE_COL,
            TIME_COL,
            TARGET,
        ]
    ].copy()

    if "load_valid" in test.columns:

        pred_part["load_valid"] = (
            test["load_valid"].to_numpy()
        )

    pred_part["Prediction"] = pred
    pred_part["Month"] = month

    rolling_pred_parts.append(
        pred_part
    )


monthly_result = pd.DataFrame(
    monthly_rows
)

rolling_predictions = pd.concat(
    rolling_pred_parts,
    ignore_index=True
)


# ============================================================
# 8. 2026 总结
# ============================================================

monthly_mean_mape = float(
    monthly_result["MAPE"].mean()
)


score_mask_all = (
    rolling_predictions[TARGET].notna()
)

if "load_valid" in rolling_predictions.columns:

    score_mask_all &= (
        rolling_predictions["load_valid"]
        ==
        1
    )


overall_2026 = calc_metrics(

    rolling_predictions.loc[
        score_mask_all,
        TARGET
    ],

    rolling_predictions.loc[
        score_mask_all,
        "Prediction"
    ],
)


print("\n" + "=" * 100)
print("2026 FINAL SUMMARY")
print("=" * 100)

print(
    monthly_result[
        [
            "Month",
            "Scored_Rows",
            "MAPE",
            "WMAPE",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print(
    f"\nMonthly Mean MAPE = "
    f"{monthly_mean_mape:.6f}%"
)

print(
    f"Pooled MAPE       = "
    f"{overall_2026['MAPE']:.6f}%"
)

print(
    f"Pooled WMAPE      = "
    f"{overall_2026['WMAPE']:.6f}%"
)

print(
    f"Scored points     = "
    f"{int(score_mask_all.sum())}"
)


# ============================================================
# 9. 和正式 reference 比较
# ============================================================

REF_2026_MONTHLY = 2.546427
REF_2026_POOLED = 2.555486
REF_2026_SCORED = 22271


print("\n" + "=" * 100)
print("REPRODUCTION CHECK")
print("=" * 100)

print(
    f"2026 Monthly Mean:"
    f"\n  reference = {REF_2026_MONTHLY:.6f}%"
    f"\n  reproduced= {monthly_mean_mape:.6f}%"
    f"\n  diff      = "
    f"{monthly_mean_mape-REF_2026_MONTHLY:+.6f} pp"
)

print(
    f"\n2026 Pooled MAPE:"
    f"\n  reference = {REF_2026_POOLED:.6f}%"
    f"\n  reproduced= {overall_2026['MAPE']:.6f}%"
    f"\n  diff      = "
    f"{overall_2026['MAPE']-REF_2026_POOLED:+.6f} pp"
)

print(
    f"\nScored points:"
    f"\n  reference = {REF_2026_SCORED}"
    f"\n  reproduced= {int(score_mask_all.sum())}"
)


# ============================================================
# 10. 保存 Release Reproduction
# ============================================================

cv_result.to_csv(
    OUT_DIR / "lightgbm_2025_cv.csv",
    index=False
)

cv_predictions.to_csv(
    OUT_DIR / "lightgbm_2025_predictions.csv",
    index=False
)

monthly_result.to_csv(
    OUT_DIR / "lightgbm_2026_monthly.csv",
    index=False
)

rolling_predictions.to_csv(
    OUT_DIR / "lightgbm_2026_predictions.csv",
    index=False
)


summary = {

    "model":
        "LightGBM V14 Final",

    "feature_count":
        len(FEATURES),

    "best_iterations":
        best_iterations,

    "rolling_n_estimators":
        final_n_estimators,

    "cv_mean_mape":
        cv_mean_mape,

    "cv_pooled_mape":
        pooled_cv["MAPE"],

    "2026_monthly_mean_mape":
        monthly_mean_mape,

    "2026_pooled_mape":
        overall_2026["MAPE"],

    "2026_pooled_wmape":
        overall_2026["WMAPE"],

    "2026_scored_points":
        int(score_mask_all.sum()),

    "reference": {

        "cv_mean_mape":
            REF_CV_MEAN,

        "rolling_n_estimators":
            REF_N_ESTIMATORS,

        "2026_monthly_mean_mape":
            REF_2026_MONTHLY,

        "2026_pooled_mape":
            REF_2026_POOLED,

        "2026_scored_points":
            REF_2026_SCORED,
    }
}


with open(
    OUT_DIR / "reproduction_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n" + "=" * 100)
print("✅ LIGHTGBM RELEASE REPRODUCTION DONE")
print("=" * 100)
print("Saved to:")
print(OUT_DIR)