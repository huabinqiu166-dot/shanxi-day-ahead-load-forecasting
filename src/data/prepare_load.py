"""
Build the unified Shanxi load table used by the forecasting models.

Input
-----
data/processed/shanxi_system_load.csv

Required columns:
    timestamp
    system_load_mw

Output
------
data/derived/unified_load.pkl

The preprocessing reproduces the load-cleaning procedure used in the
final experiments:

1. NaN or non-positive load values are marked invalid.
2. Historical same-slot templates use only prior valid real observations.
3. Template windows expand from 7 to 14 to 28 days when required.
4. Completely missing days use the historical template directly.
5. Partially missing days use a day-level median ratio correction.

Filled observations are never recursively reused as real historical values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_END_DATE = pd.Timestamp("2026-08-20")


# ============================================================
# Time construction
# ============================================================

def build_calendar_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamps to target_date / slot representation.

    A target day contains 96 points:

        slot 1   = D 00:15
        ...
        slot 95  = D 23:45
        slot 96  = D+1 00:00
    """

    out = df.copy()

    out["timestamp"] = pd.to_datetime(
        out["timestamp"]
    )

    out = (
        out
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    minute_of_day = (
        out["timestamp"].dt.hour * 60
        +
        out["timestamp"].dt.minute
    )

    is_midnight = (
        minute_of_day == 0
    )

    # Normal timestamps belong to their calendar day.
    out["目标日期"] = (
        out["timestamp"]
        .dt.normalize()
    )

    # 00:00 is slot 96 of the previous target day.
    out.loc[
        is_midnight,
        "目标日期"
    ] = (
        out.loc[
            is_midnight,
            "目标日期"
        ]
        -
        pd.Timedelta(days=1)
    )

    out["slot"] = (
        minute_of_day // 15
    ).astype(int)

    out.loc[
        is_midnight,
        "slot"
    ] = 96

    out["时刻"] = (
        out["timestamp"]
        .dt.strftime("%H:%M")
    )

    # Historical project convention:
    # slot 96 belongs to target day D but its real timestamp is D+1 00:00.
    # Keep the display label as "24:00" for exact reproducibility.
    out.loc[
        out["slot"] == 96,
        "时刻"
    ] = "24:00"

    return out


# ============================================================
# Historical same-slot template
# ============================================================

def historical_slot_template(
    data: pd.DataFrame,
    target_day: pd.Timestamp,
    slot: int,
):
    """
    Estimate the historical template for one slot.

    Windows are checked in this order:

        7 days -> 14 days -> 28 days

    Only prior REAL valid observations (load_raw) are allowed.
    Filled values are never reused.
    """

    target_day = pd.Timestamp(
        target_day
    ).normalize()

    for window_days in (
        7,
        14,
        28,
    ):

        start_day = (
            target_day
            -
            pd.Timedelta(
                days=window_days
            )
        )

        mask = (
            (data["目标日期"] >= start_day)
            &
            (data["目标日期"] < target_day)
            &
            (data["slot"] == slot)
            &
            (data["load_valid"] == 1)
            &
            data["load_raw"].notna()
        )

        values = (
            data.loc[
                mask,
                "load_raw"
            ]
            .astype(float)
            .to_numpy()
        )

        if len(values) > 0:

            return (
                float(
                    np.median(values)
                ),
                window_days,
            )

    return (
        np.nan,
        np.nan,
    )


# ============================================================
# Main preprocessing
# ============================================================

def prepare_load(
    input_path: Path,
    model_end_date: pd.Timestamp = MODEL_END_DATE,
) -> pd.DataFrame:
    """
    Build the exact unified load representation used by the models.
    """

    raw = pd.read_csv(
        input_path
    )

    required = {
        "timestamp",
        "system_load_mw",
    }

    missing = (
        required
        -
        set(raw.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Time / target-day structure
    # --------------------------------------------------------

    raw = build_calendar_index(
        raw
    )

    # Keep target dates through the frozen project cutoff.
    raw = (
        raw[
            raw["目标日期"]
            <= model_end_date
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Every target day must contain 96 points.
    day_counts = (
        raw
        .groupby("目标日期")
        .size()
    )

    bad_days = day_counts[
        day_counts != 96
    ]

    if len(bad_days) > 0:

        raise ValueError(
            "Target days without exactly 96 points:\n"
            + bad_days.to_string()
        )

    if raw["slot"].min() != 1:
        raise ValueError(
            "Minimum slot is not 1."
        )

    if raw["slot"].max() != 96:
        raise ValueError(
            "Maximum slot is not 96."
        )

    # --------------------------------------------------------
    # Original / validity / raw values
    # --------------------------------------------------------

    raw[
        "系统负荷实际值(MW)"
    ] = raw[
        "system_load_mw"
    ]

    raw[
        "load_original"
    ] = raw[
        "system_load_mw"
    ]

    raw[
        "load_valid"
    ] = (
        raw[
            "load_original"
        ].notna()
        &
        (
            raw[
                "load_original"
            ] > 0
        )
    ).astype(int)

    raw[
        "load_raw"
    ] = raw[
        "load_original"
    ].where(
        raw[
            "load_valid"
        ] == 1,
        np.nan,
    )

    # --------------------------------------------------------
    # Fill-related columns
    # --------------------------------------------------------

    raw[
        "load_filled"
    ] = raw[
        "load_raw"
    ]

    raw[
        "fill_template"
    ] = np.nan

    raw[
        "fill_window_days"
    ] = np.nan

    raw[
        "day_adjust_ratio"
    ] = 1.0

    # --------------------------------------------------------
    # Process target days chronologically
    # --------------------------------------------------------

    target_days = (
        raw["目标日期"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for target_day in target_days:

        day_mask = (
            raw["目标日期"]
            == target_day
        )

        day_rows = (
            raw.loc[
                day_mask
            ]
            .copy()
        )

        invalid_mask = (
            day_rows[
                "load_valid"
            ] == 0
        )

        n_invalid = int(
            invalid_mask.sum()
        )

        # No missing / invalid points.
        if n_invalid == 0:
            continue

        # ----------------------------------------------------
        # Historical templates for all 96 slots
        # ----------------------------------------------------

        templates = {}
        template_windows = {}

        for slot in range(
            1,
            97,
        ):

            template, used_window = (
                historical_slot_template(
                    raw,
                    target_day,
                    slot,
                )
            )

            templates[
                slot
            ] = template

            template_windows[
                slot
            ] = used_window

        # ----------------------------------------------------
        # Day-level adjustment ratio
        # ----------------------------------------------------

        valid_day_rows = (
            day_rows[
                day_rows[
                    "load_valid"
                ] == 1
            ]
            .copy()
        )

        # Entire day missing:
        # use historical template directly.
        if len(valid_day_rows) == 0:

            day_ratio = 1.0

        else:

            ratios = []

            for _, row in (
                valid_day_rows.iterrows()
            ):

                slot = int(
                    row["slot"]
                )

                template = (
                    templates[
                        slot
                    ]
                )

                actual = (
                    row[
                        "load_raw"
                    ]
                )

                if (
                    pd.notna(template)
                    and
                    np.isfinite(template)
                    and
                    template > 0
                    and
                    pd.notna(actual)
                    and
                    np.isfinite(actual)
                    and
                    actual > 0
                ):

                    ratios.append(
                        float(actual)
                        /
                        float(template)
                    )

            if len(ratios) > 0:

                day_ratio = float(
                    np.median(
                        ratios
                    )
                )

            else:

                day_ratio = 1.0

        # ----------------------------------------------------
        # Fill only invalid points
        # ----------------------------------------------------

        invalid_indices = (
            day_rows.index[
                invalid_mask
            ]
        )

        for idx in invalid_indices:

            slot = int(
                raw.at[
                    idx,
                    "slot"
                ]
            )

            template = (
                templates[
                    slot
                ]
            )

            used_window = (
                template_windows[
                    slot
                ]
            )

            if (
                pd.isna(template)
                or
                not np.isfinite(
                    template
                )
            ):

                raise RuntimeError(
                    "Unable to build historical "
                    f"template for {target_day} "
                    f"slot={slot}"
                )

            filled = (
                float(template)
                *
                float(day_ratio)
            )

            raw.at[
                idx,
                "load_filled"
            ] = filled

            raw.at[
                idx,
                "fill_template"
            ] = template

            raw.at[
                idx,
                "fill_window_days"
            ] = used_window

            raw.at[
                idx,
                "day_adjust_ratio"
            ] = day_ratio

    # --------------------------------------------------------
    # Final integrity checks
    # --------------------------------------------------------

    if raw[
        "load_filled"
    ].isna().any():

        raise RuntimeError(
            "load_filled still contains NaN."
        )

    output_columns = [
        "目标日期",
        "slot",
        "时刻",
        "timestamp",
        "系统负荷实际值(MW)",
        "load_original",
        "load_valid",
        "load_raw",
        "load_filled",
        "fill_template",
        "fill_window_days",
        "day_adjust_ratio",
    ]

    return (
        raw[
            output_columns
        ]
        .sort_values(
            [
                "目标日期",
                "slot",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Command line entry point
# ============================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    default_input = (
        project_root
        / "data"
        / "processed"
        / "shanxi_system_load.csv"
    )

    default_output = (
        project_root
        / "data"
        / "derived"
        / "unified_load.pkl"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the unified Shanxi load "
            "dataset used by the forecasting models."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Public Shanxi system load CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output unified load pickle.",
    )

    parser.add_argument(
        "--model-end-date",
        type=str,
        default="2026-08-20",
        help="Final target date to retain.",
    )

    args = parser.parse_args()

    model_end_date = pd.Timestamp(
        args.model_end_date
    )

    result = prepare_load(
        input_path=args.input,
        model_end_date=model_end_date,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_pickle(
        args.output
    )

    invalid_points = int(
        (
            result[
                "load_valid"
            ] == 0
        ).sum()
    )

    print(
        "Unified load created successfully."
    )

    print(
        "Output:",
        args.output
    )

    print(
        "Shape:",
        result.shape
    )

    print(
        "Target dates:",
        result["目标日期"].min(),
        "->",
        result["目标日期"].max()
    )

    print(
        "Invalid raw points:",
        invalid_points
    )


if __name__ == "__main__":
    main()
