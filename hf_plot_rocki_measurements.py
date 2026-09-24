"""
Boxplots of ROCKi staining measurements.

One figure per channel. Location is on the x-axis, treatment is hue
(boxes and points share a colour). Measurements from the same embryo
are connected across locations with gray lines.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────
CSV_PATH = r"D:\Ruoheng_Li\20260831-dispbead+ROCKibead+WT_staining\ROCKi-measurements.csv"
OUTPUT_DIR = os.path.dirname(CSV_PATH)
METRIC = "median"

LOCATION_ORDER = ["patch", "anterior"]
TREATMENT_ORDER = ["ctrl bead", "ROCKi bead", "ctrl plate", "ROCKi plate"]
TREATMENT_COLORS = {
    "ctrl bead": "#1f77b4",
    "ROCKi bead": "#d62728",
    "ctrl plate": "#aec7e8",
    "ROCKi plate": "#ff9896",
}
BOX_SPAN = 0.8
# ─────────────────────────────────────────────


def hue_offsets(hue_order: list[str], span: float = BOX_SPAN) -> dict[str, float]:
    n_hue = len(hue_order)
    dodge = span / n_hue
    return {
        treatment: (i - (n_hue - 1) / 2) * dodge
        for i, treatment in enumerate(hue_order)
    }


def plot_channel(ax, data: pd.DataFrame, channel: str, metric: str = METRIC) -> None:
    plot_data = data.loc[data["channel"] == channel].copy()
    location_pos = {loc: i for i, loc in enumerate(LOCATION_ORDER)}
    offsets = hue_offsets(TREATMENT_ORDER)

    sns.boxplot(
        x="location",
        y=metric,
        hue="treatment",
        data=plot_data,
        order=LOCATION_ORDER,
        hue_order=TREATMENT_ORDER,
        palette=TREATMENT_COLORS,
        linewidth=1.5,
        boxprops=dict(edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
        flierprops=dict(markerfacecolor="black", markeredgecolor="black"),
        width=BOX_SPAN,
        dodge=True,
        showfliers=False,
        ax=ax,
    )

    for embryo, embryo_df in plot_data.groupby("embryo"):
        embryo_df = embryo_df.dropna(subset=[metric]).sort_values(
            "location",
            key=lambda s: s.map(location_pos),
        )
        if len(embryo_df) < 2:
            continue
        treatment = embryo_df["treatment"].iloc[0]
        xs = [
            location_pos[loc] + offsets[treatment]
            for loc in embryo_df["location"]
        ]
        ax.plot(
            xs,
            embryo_df[metric].to_numpy(),
            color="gray",
            alpha=0.6,
            zorder=2,
        )

    sns.stripplot(
        x="location",
        y=metric,
        hue="treatment",
        data=plot_data,
        order=LOCATION_ORDER,
        hue_order=TREATMENT_ORDER,
        palette=TREATMENT_COLORS,
        dodge=True,
        jitter=False,
        size=6,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
        legend=False,
        zorder=3,
    )

    handles, labels = ax.get_legend_handles_labels()
    n_hue = len(TREATMENT_ORDER)
    ax.legend(handles[:n_hue], labels[:n_hue], title="treatment", frameon=False)
    ax.set_xlabel("location")
    ax.set_ylabel(f"{metric} intensity")
    ax.set_title(channel)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    channels = [c for c in df["channel"].dropna().unique() if pd.notna(c)]
    # Keep a stable order: actin then pMLC if both present
    preferred = ["actin", "pMLC"]
    channels = [c for c in preferred if c in channels] + [
        c for c in channels if c not in preferred
    ]

    sns.set_theme(style="white")
    for channel in channels:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_channel(ax, df, channel)
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"ROCKi-measurements_{channel}_{METRIC}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
