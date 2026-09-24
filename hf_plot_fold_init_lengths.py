"""
Extract length metrics at each embryo's fold_init timepoint and plot boxplots.

Embryos are identified by INDEX_COLUMNS. If a 'treatment' column is present,
treatments are plotted as separate boxes.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────
OFFSET_CSV = "D:\\Ruoheng_Li\\20260830-dispbead\\offset.csv"
LENGTHS_CSV = "D:\\Ruoheng_Li\\20260830-dispbead\\lengths.csv"
OUTPUT_PNG = "D:\\Ruoheng_Li\\20260830-dispbead\\fold_init_lengths_boxplot.png"

INDEX_COLUMNS = ["date", "embryo"]  # columns that identify one embryo
TREATMENT_COLUMN = "treatment"
METRICS = ("l_notochord", "l_aNNE")

SCALE = 0.859
# ─────────────────────────────────────────────


def as_key_tuple(keys) -> tuple:
    if isinstance(keys, tuple):
        return keys
    return (keys,)


def fold_init_column(offset: pd.DataFrame) -> str:
    for column in offset.columns:
        if column.replace(" ", "_").lower() == "fold_init":
            return column
    raise KeyError("Could not find 'fold_init' column in offset.csv")


def has_treatment(df: pd.DataFrame) -> bool:
    return TREATMENT_COLUMN in df.columns and df[TREATMENT_COLUMN].notna().any()


def match_index(df: pd.DataFrame, key_values) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for column, value in zip(INDEX_COLUMNS, as_key_tuple(key_values)):
        mask &= df[column] == value
    return df.loc[mask]


def treatment_for_embryo(offset_row: pd.Series, embryo_lengths: pd.DataFrame):
    if TREATMENT_COLUMN in offset_row.index and pd.notna(offset_row[TREATMENT_COLUMN]):
        return offset_row[TREATMENT_COLUMN]
    if TREATMENT_COLUMN in embryo_lengths.columns:
        treatments = embryo_lengths[TREATMENT_COLUMN].dropna()
        if not treatments.empty:
            return treatments.iloc[0]
    return np.nan


def value_at_time(embryo_lengths: pd.DataFrame, timepoint: float, column: str) -> float:
    if pd.isna(timepoint):
        return np.nan

    timepoint = float(timepoint)
    if timepoint == int(timepoint):
        row = embryo_lengths.loc[embryo_lengths["time"] == int(timepoint), column]
        return row.iloc[0] if not row.empty else np.nan

    lower = int(np.floor(timepoint))
    upper = int(np.ceil(timepoint))
    lower_val = embryo_lengths.loc[embryo_lengths["time"] == lower, column]
    upper_val = embryo_lengths.loc[embryo_lengths["time"] == upper, column]
    if lower_val.empty or upper_val.empty:
        return np.nan

    lower_value = lower_val.iloc[0]
    upper_value = upper_val.iloc[0]
    if pd.isna(lower_value) or pd.isna(upper_value):
        return np.nan
    return (lower_value + upper_value) / 2


def extract_fold_init_lengths(offset: pd.DataFrame, lengths: pd.DataFrame) -> pd.DataFrame:
    fold_col = fold_init_column(offset)
    missing_offset = [c for c in INDEX_COLUMNS if c not in offset.columns]
    missing_lengths = [c for c in INDEX_COLUMNS + ["time"] if c not in lengths.columns]
    if missing_offset:
        raise KeyError(f"Missing index columns in {OFFSET_CSV}: {missing_offset}")
    if missing_lengths:
        raise KeyError(f"Missing columns in {LENGTHS_CSV}: {missing_lengths}")

    rows = []
    for _, embryo_offset in offset.iterrows():
        key_values = tuple(embryo_offset[c] for c in INDEX_COLUMNS)
        fold_init = embryo_offset[fold_col]
        embryo_lengths = match_index(lengths, key_values)

        row = {col: val for col, val in zip(INDEX_COLUMNS, key_values)}
        row["fold_init"] = fold_init
        if has_treatment(offset) or has_treatment(lengths):
            row[TREATMENT_COLUMN] = treatment_for_embryo(embryo_offset, embryo_lengths)
        for metric in METRICS:
            row[metric] = value_at_time(embryo_lengths, fold_init, metric)
        rows.append(row)

    return pd.DataFrame(rows)


def print_metric_stats(data: pd.DataFrame, label: str = "") -> None:
    prefix = f"{label}: " if label else ""
    for metric in METRICS:
        values = data[metric].dropna() * SCALE
        print(f"{prefix}{metric}: mean={values.mean():.2f}, std={values.std():.2f}, n={len(values)}")


def plot_metric_boxplot(ax, data: pd.DataFrame, metric: str, scale=SCALE) -> None:
    plot_data = data.dropna(subset=[metric]).copy()
    plot_data[metric] = plot_data[metric] * scale
    treated = has_treatment(plot_data)

    box_kwargs = dict(
        y=metric,
        data=plot_data,
        ax=ax,
        linewidth=1.5,
        boxprops=dict(facecolor="white", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
        flierprops=dict(markerfacecolor="black", markeredgecolor="black"),
        width=0.4,
    )
    strip_kwargs = dict(
        y=metric,
        data=plot_data,
        ax=ax,
        color="black",
        alpha=0.7,
        jitter=0.15,
        size=5,
    )

    if treated:
        treatments = sorted(plot_data[TREATMENT_COLUMN].dropna().unique(), key=str)
        box_kwargs["x"] = TREATMENT_COLUMN
        box_kwargs["order"] = treatments
        strip_kwargs["x"] = TREATMENT_COLUMN
        strip_kwargs["order"] = treatments
        counts = plot_data.groupby(TREATMENT_COLUMN, dropna=False)[metric].count()
        n_label = ", ".join(f"{t} n={n}" for t, n in counts.items())
    else:
        n_label = f"n={len(plot_data)}"

    sns.boxplot(**box_kwargs)
    sns.stripplot(**strip_kwargs)
    ax.set_ylim(0,2000)
    ax.set_xlabel(TREATMENT_COLUMN if treated else "")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} at fold init ({n_label})")


offset = pd.read_csv(OFFSET_CSV)
lengths = pd.read_csv(LENGTHS_CSV)
fold_init_data = extract_fold_init_lengths(offset, lengths)

print("Mean and std at fold init (scaled):")
if has_treatment(fold_init_data):
    for treatment, sub in fold_init_data.groupby(TREATMENT_COLUMN, dropna=False, sort=True):
        print_metric_stats(sub, label=str(treatment))
else:
    print_metric_stats(fold_init_data)

n_treatments = (
    fold_init_data[TREATMENT_COLUMN].nunique(dropna=True)
    if has_treatment(fold_init_data)
    else 1
)
sns.set_theme(style="white")
fig, axes = plt.subplots(1, len(METRICS), figsize=(max(4, 3 * n_treatments) * len(METRICS) / 2, 5))
axes = np.atleast_1d(axes)

for ax, metric in zip(axes, METRICS):
    plot_metric_boxplot(ax, fold_init_data, metric)

fig.suptitle("Lengths at fold init timepoint")
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.show()
