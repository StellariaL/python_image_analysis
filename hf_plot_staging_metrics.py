import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

DATA_CSV = os.path.join("data", "lengths.csv")
OUTPUT_PNG = os.path.join("data", "lengths_loess.png")
OUTPUT_SLOPE_PNG = os.path.join("data", "lengths_residual_slope.png")

KEY_COLUMNS = ("embryo", "time", "aligned_time")
EXCLUDE_COLUMNS = [
    "l_aNNE","l_streak"
]
LOESS_FRAC = 0.4
SUBPLOT_COLS = 3
MIN_ABS_SLOPE = 1e-6


def get_data_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in KEY_COLUMNS and col not in EXCLUDE_COLUMNS
    ]


def loess_fit_and_residual_std(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = LOESS_FRAC,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 3:
        return None, None, None

    fitted_at_points = lowess(
        y_clean,
        x_clean,
        frac=frac,
        return_sorted=False,
    )
    residual_std = float(np.std(y_clean - fitted_at_points))
    unique_x, inverse_indices = np.unique(x_clean, return_inverse=True)
    
    unique_fitted_y = np.bincount(
        inverse_indices, weights=fitted_at_points
    ) / np.bincount(inverse_indices)

    return unique_x, unique_fitted_y, residual_std


def residual_std_over_local_slope(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    residual_std: float,
) -> np.ndarray:
    if len(fit_x) < 2:
        return np.array([])

    dx = np.diff(fit_x)
    dy = np.diff(fit_y)
    segment_slope = np.divide(
        dy,
        dx,
        out=np.zeros_like(dy),
        where=np.abs(dx) > MIN_ABS_SLOPE,
    )

    local_slope = np.zeros_like(fit_x)
    local_slope[0] = segment_slope[0]
    local_slope[-1] = segment_slope[-1]
    if len(segment_slope) > 1:
        local_slope[1:-1] = 0.5 * (segment_slope[:-1] + segment_slope[1:])

    denom = np.maximum(np.abs(local_slope), MIN_ABS_SLOPE)
    return residual_std / denom

def local_slope(
    fit_x: np.ndarray,
    fit_y: np.ndarray
) -> np.ndarray:
    if len(fit_x) < 2:
        return np.array([])

    dx = np.diff(fit_x)
    dy = np.diff(fit_y)
    segment_slope = np.divide(
        dy,
        dx,
        out=np.zeros_like(dy),
        where=np.abs(dx) > MIN_ABS_SLOPE,
    )
    local_slope = np.zeros_like(fit_x)
    local_slope[0] = segment_slope[0]
    local_slope[-1] = segment_slope[-1]
    if len(segment_slope) > 1:
        local_slope[1:-1] = 0.5 * (segment_slope[:-1] + segment_slope[1:])

    denom = np.maximum(np.abs(local_slope), MIN_ABS_SLOPE)
    return denom


df = pd.read_csv(DATA_CSV)
data_columns = get_data_columns(df)

fit_results: dict[str, dict] = {}
for column in data_columns:
    fit_x, fit_y, residual_std = loess_fit_and_residual_std(
        df["aligned_time"].to_numpy(),
        df[column].to_numpy(),
    )
    fit_results[column] = {
        "fit_x": fit_x,
        "fit_y": fit_y,
        "residual_std": residual_std,
    }

valid_results = [
    (col, result["residual_std"])
    for col, result in fit_results.items()
    if result["residual_std"] is not None
]
ranked = sorted(valid_results, key=lambda item: item[1])
smallest_column = ranked[0][0] if ranked else None

print("Residual standard deviation ranking (smallest to largest):")
for rank, (column, residual_std) in enumerate(ranked, start=1):
    marker = "  <-- smallest" if column == smallest_column else ""
    print(f"  {rank:2d}. {column}: {residual_std:.4f}{marker}")

if smallest_column is not None:
    print(f"\nSmallest residual std: {smallest_column} ({ranked[0][1]:.4f})")

n_cols = SUBPLOT_COLS
n_rows = math.ceil(len(data_columns) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharex=True)
axes = np.atleast_1d(axes).ravel()

for ax, column in zip(axes, data_columns):
    result = fit_results[column]
    fit_x = result["fit_x"]
    fit_y = result["fit_y"]
    residual_std = result["residual_std"]

    for embryo in sorted(df["embryo"].unique()):
        sub = df[df["embryo"] == embryo].sort_values("aligned_time")
        valid = sub[column].notna()
        if not valid.any():
            continue
        ax.plot(
            sub.loc[valid, "aligned_time"],
            sub.loc[valid, column],
            marker="o",
            linewidth=1.0,
            markersize=3,
            alpha=0.45,
            label=f"embryo {embryo}",
        )

    if fit_x is not None:
        ax.plot(fit_x, fit_y, color="black", linewidth=2.5, label="LOESS fit")

    std_label = f"{residual_std:.2f}" if residual_std is not None else "n/a"
    title = f"{column}\n(residual std = {std_label})"
    if column == smallest_column:
        title += " *"
        for spine in ax.spines.values():
            spine.set_edgecolor("green")
            spine.set_linewidth(2.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)

for ax in axes[len(data_columns) :]:
    ax.set_visible(False)

for ax in axes[-n_cols:]:
    if ax.get_visible():
        ax.set_xlabel("aligned_time")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Embryo",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left",
    fontsize=7,
)
fig.suptitle(
    f"LOESS fits vs aligned time (* = smallest residual std: {smallest_column})",
    y=1.02,
)
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")


fig_ratio, ax_ratio = plt.subplots(figsize=(10, 6))
for column in data_columns:
    result = fit_results[column]
    fit_x = result["fit_x"]
    fit_y = result["fit_y"]
    residual_std = result["residual_std"]
    if fit_x is None or residual_std is None:
        continue

    ratio = residual_std_over_local_slope(fit_x, fit_y, residual_std)
    ax_ratio.plot(
        fit_x,
        ratio,
        linewidth=2,
        label=column,
    )

ax_ratio.set_xlabel("aligned_time")
ax_ratio.set_ylabel("residual std / |local LOESS slope|")
ax_ratio.set_title("Residual std normalized by local LOESS slope")
ax_ratio.grid(True, alpha=0.3)
ax_ratio.legend(title="Column", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig_ratio.tight_layout()
fig_ratio.savefig(OUTPUT_SLOPE_PNG, dpi=300, bbox_inches="tight")
plt.show()
