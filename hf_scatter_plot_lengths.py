"""
Scatter one numeric column against another, with one series per unique
combination of index columns.

If a 'treatment' column is present, each treatment is distinguished either by
marker shape (colour then identifies the series) or by colour.
Optionally overlay a least-squares linear fit, pooled or per treatment.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────
DATA_CSV = "D:\\Ruoheng_Li\\20260910-dispbead_ant\\lengths-0910+0911.csv"
OUTPUT_DIR = "D:\\Ruoheng_Li\\20260910-dispbead_ant"

INDEX_COLUMNS = ["embryo","date"]   # columns that define one series
X_COLUMN = "l_notochord"
Y_COLUMN = "l_embryo"
X_SCALE = 4.037630718   # multiply x values by this before plotting
Y_SCALE = 4.037630718   # multiply y values by this before plotting
TREATMENT_COLUMN = "treatment"
TREATMENT_STYLE = "color"   # "marker": colour per series, marker per treatment
                             # "color":  colour per treatment, same marker throughout

FIT_LINE = True              # overlay a least-squares linear fit
FIT_BY_TREATMENT = True     # fit each treatment separately instead of pooling all points

TREATMENT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TREATMENT_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]
DEFAULT_MARKER = "o"
# Used to tell per-treatment fits apart when treatments are shown as markers.
FIT_LINESTYLES = ["--", ":", "-.", (0, (5, 1, 1, 1, 1, 1))]
# ─────────────────────────────────────────────

# Legends are stacked to the right of the axes, in creation order.
LEGEND_SLOTS = [
    ((1.05, 1), "upper left"),
    ((1.05, 0), "lower left"),
    ((1.05, 0.5), "center left"),
]


def as_key_tuple(keys) -> tuple:
    if isinstance(keys, tuple):
        return keys
    return (keys,)


def series_label(keys, columns: list[str]) -> str:
    return ", ".join(f"{col} {val}" for col, val in zip(columns, as_key_tuple(keys)))


def treatment_style_map(df: pd.DataFrame) -> dict | None:
    """Map each treatment to a marker or a colour, following TREATMENT_STYLE."""
    if TREATMENT_COLUMN not in df.columns:
        return None
    treatments = sorted(df[TREATMENT_COLUMN].dropna().unique(), key=str)
    if not treatments:
        return None
    styles = TREATMENT_MARKERS if TREATMENT_STYLE == "marker" else TREATMENT_COLORS
    return {t: styles[i % len(styles)] for i, t in enumerate(treatments)}


def treatment_legend_handles(style_map: dict) -> list[Line2D]:
    handles = []
    for treatment, style in style_map.items():
        color = "black" if TREATMENT_STYLE == "marker" else style
        marker = style if TREATMENT_STYLE == "marker" else DEFAULT_MARKER
        handles.append(
            Line2D([0], [0], color=color, marker=marker, linestyle="None", label=str(treatment))
        )
    return handles


def add_linear_fit(ax, x: pd.Series, y: pd.Series, color, linestyle, label_prefix: str):
    """Draw the least-squares fit of y on x, returning its legend handle."""
    if len(x) < 2 or x.nunique() < 2:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    total = y - y.mean()
    r2 = 1 - residuals.pow(2).sum() / total.pow(2).sum() if total.any() else float("nan")

    x_ends = np.array([x.min(), x.max()])
    (line,) = ax.plot(
        x_ends,
        slope * x_ends + intercept,
        color=color,
        linestyle=linestyle,
        linewidth=1.5,
        label=f"{label_prefix}y = {slope:.3g}x {intercept:+.3g}  (R² = {r2:.3f})",
    )
    return line


df = pd.read_csv(DATA_CSV)

missing = [c for c in INDEX_COLUMNS + [X_COLUMN, Y_COLUMN] if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in {DATA_CSV}: {missing}")

if TREATMENT_STYLE not in {"marker", "color"}:
    raise ValueError(f"TREATMENT_STYLE must be 'marker' or 'color', got {TREATMENT_STYLE!r}")

df[X_COLUMN] = df[X_COLUMN] * X_SCALE
df[Y_COLUMN] = df[Y_COLUMN] * Y_SCALE

style_map = treatment_style_map(df)

plt.figure(figsize=(10, 6))

series_handles = []
plotted_y = []

for keys, sub in df.groupby(INDEX_COLUMNS, sort=True, dropna=False):
    sub = sub.sort_values(X_COLUMN)
    valid = sub[X_COLUMN].notna() & sub[Y_COLUMN].notna()
    if not valid.any():
        continue

    plotted_y.append(sub.loc[valid, Y_COLUMN])

    plot_kwargs = {"marker": DEFAULT_MARKER, "linestyle": "None"}
    if style_map is not None:
        treatments = sub[TREATMENT_COLUMN].dropna()
        if not treatments.empty:
            style = style_map[treatments.iloc[0]]
            if TREATMENT_STYLE == "marker":
                plot_kwargs["marker"] = style
            else:
                plot_kwargs["color"] = style

    (handle,) = plt.plot(
        sub.loc[valid, X_COLUMN],
        sub.loc[valid, Y_COLUMN],
        label=series_label(keys, INDEX_COLUMNS),
        **plot_kwargs,
    )
    series_handles.append(handle)

ax = plt.gca()

fit_handles = []
if FIT_LINE:
    valid = df[X_COLUMN].notna() & df[Y_COLUMN].notna()
    fit_df = df[valid]
    if FIT_BY_TREATMENT and style_map is not None:
        for i, (treatment, sub) in enumerate(fit_df.groupby(TREATMENT_COLUMN, sort=True)):
            # Whichever channel the scatter does not use identifies the fits.
            if TREATMENT_STYLE == "color":
                color, linestyle = style_map[treatment], "--"
            else:
                color = "black"
                linestyle = FIT_LINESTYLES[i % len(FIT_LINESTYLES)]
            handle = add_linear_fit(
                ax, sub[X_COLUMN], sub[Y_COLUMN], color, linestyle, f"{treatment}: "
            )
            if handle is not None:
                fit_handles.append(handle)
    else:
        handle = add_linear_fit(ax, fit_df[X_COLUMN], fit_df[Y_COLUMN], "black", "--", "")
        if handle is not None:
            fit_handles.append(handle)

plt.xlabel(X_COLUMN)
plt.ylabel(Y_COLUMN)
plt.title(f"{Y_COLUMN} vs {X_COLUMN}")

legend_specs = []
# In colour mode the series all share their treatment's colour, so listing them
# individually would not tell them apart.
if style_map is None or TREATMENT_STYLE == "marker":
    legend_specs.append((series_handles, " / ".join(INDEX_COLUMNS)))
if style_map is not None:
    legend_specs.append((treatment_legend_handles(style_map), TREATMENT_COLUMN))
if fit_handles:
    legend_specs.append((fit_handles, "linear fit"))

for i, ((handles, title), (anchor, loc)) in enumerate(zip(legend_specs, LEGEND_SLOTS)):
    legend = ax.legend(handles=handles, title=title, bbox_to_anchor=anchor, loc=loc)
    if i < len(legend_specs) - 1:
        ax.add_artist(legend)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_png = os.path.join(OUTPUT_DIR, f"{Y_COLUMN}_against_{X_COLUMN}_scatter.png")

plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches="tight")
print(f"Saved {output_png}")
plt.show()
