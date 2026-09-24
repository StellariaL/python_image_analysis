"""
Plot one numeric column against another, with one series per unique
combination of index columns.

PLOT_TYPE may be "line" (markers connected) or "scatter" (markers only).
If a 'treatment' column is present, each treatment gets a distinct marker.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────
DATA_CSV = "D:\\Ruoheng_Li\\20260910-dispbead_ant\\lengths-0910+0911.csv"
OUTPUT_DIR = "D:\\Ruoheng_Li\\20260910-dispbead_ant"

INDEX_COLUMNS = ["embryo","date"]   # columns that define one series
X_COLUMN = "time"
Y_COLUMN = "l_notochord_diff"
X_SCALE = 1   # multiply x values by this before plotting
Y_SCALE = 4.037630718   # multiply y values by this before plotting
PLOT_TYPE = "line"      # "line" or "scatter"
TREATMENT_COLUMN = "treatment"

TREATMENT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# ─────────────────────────────────────────────


def as_key_tuple(keys) -> tuple:
    if isinstance(keys, tuple):
        return keys
    return (keys,)


def series_label(keys, columns: list[str]) -> str:
    return ", ".join(f"{col} {val}" for col, val in zip(columns, as_key_tuple(keys)))


def treatment_marker_map(df: pd.DataFrame) -> dict | None:
    if TREATMENT_COLUMN not in df.columns:
        return None
    treatments = sorted(df[TREATMENT_COLUMN].dropna().unique(), key=str)
    if not treatments:
        return None
    return {
        t: TREATMENT_MARKERS[i % len(TREATMENT_MARKERS)]
        for i, t in enumerate(treatments)
    }


df = pd.read_csv(DATA_CSV)

missing = [c for c in INDEX_COLUMNS + [X_COLUMN, Y_COLUMN] if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in {DATA_CSV}: {missing}")

if PLOT_TYPE not in {"line", "scatter"}:
    raise ValueError(f"PLOT_TYPE must be 'line' or 'scatter', got {PLOT_TYPE!r}")

df[X_COLUMN] = df[X_COLUMN] * X_SCALE
df[Y_COLUMN] = df[Y_COLUMN] * Y_SCALE

marker_map = treatment_marker_map(df)

plt.figure(figsize=(10, 6))

plotted_y = []

for keys, sub in df.groupby(INDEX_COLUMNS, sort=True, dropna=False):
    sub = sub.sort_values(X_COLUMN)
    valid = sub[X_COLUMN].notna() & sub[Y_COLUMN].notna()
    if not valid.any():
        continue

    plotted_y.append(sub.loc[valid, Y_COLUMN])

    marker = "o"
    if marker_map is not None:
        treatments = sub[TREATMENT_COLUMN].dropna()
        if not treatments.empty:
            marker = marker_map[treatments.iloc[0]]

    plt.plot(
        sub.loc[valid, X_COLUMN],
        sub.loc[valid, Y_COLUMN],
        marker=marker,
        linestyle="None" if PLOT_TYPE == "scatter" else "-",
        linewidth=1.5,
        label=series_label(keys, INDEX_COLUMNS),
    )

plt.xlabel(X_COLUMN)
plt.ylabel(Y_COLUMN)
if plotted_y and pd.concat(plotted_y).min() >= 0:
    plt.ylim(bottom=0)
plt.title(f"{Y_COLUMN} vs {X_COLUMN}")

ax = plt.gca()
series_legend = ax.legend(
    title=" / ".join(INDEX_COLUMNS),
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

if marker_map is not None:
    ax.add_artist(series_legend)
    treatment_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=m,
            linestyle="None",
            label=str(t),
        )
        for t, m in marker_map.items()
    ]
    ax.legend(
        handles=treatment_handles,
        title=TREATMENT_COLUMN,
        bbox_to_anchor=(1.05, 0),
        loc="lower left",
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_png = os.path.join(OUTPUT_DIR, f"{Y_COLUMN}_against_{X_COLUMN}.png")

plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches="tight")
print(f"Saved {output_png}")
plt.show()
