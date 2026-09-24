"""
Plot AR/Ac boxplots from fit_results.csv.

Each sample (control: c*, treated: e*) is measured up to three times.
Air measurements are excluded. Individual points are shown and the
measurements from the same sample are connected with lines.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CSV_PATH = "combined-fit_results.csv"
OUTPUT_PATH = "combined-summary-diff.png"

df = pd.read_csv(CSV_PATH)
df = df[df["sample"] != "air"].copy()
df["group"] = df["sample"].str[0].map({"c": "control", "e": "treated"})
df["measurement"] = df["measurement"].astype(int)

measurements = sorted(df["measurement"].unique())
hue_order = ["control", "treated"]
n_hue = len(hue_order)
dodge = 0.8 / n_hue
group_offset = {group: (i - (n_hue - 1) / 2) * dodge for i, group in enumerate(hue_order)}
measurement_pos = {m: i for i, m in enumerate(measurements)}

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(6, 4))

sns.boxplot(
    x="measurement",
    y="exp-AR",
    hue="group",
    data=df,
    order=measurements,
    hue_order=hue_order,
    linewidth=1.5,
    boxprops=dict(facecolor="white", edgecolor="black"),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    medianprops=dict(color="black"),
    flierprops=dict(markerfacecolor="black", markeredgecolor="black"),
    width=0.4,
    dodge=True,
    ax=ax,
)

sns.stripplot(
    x="measurement",
    y="exp-AR",
    hue="group",
    data=df,
    order=measurements,
    hue_order=hue_order,
    dodge=True,
    palette={group: "black" for group in hue_order},
    alpha=0.7,
    jitter=False,
    ax=ax,
    legend=False,
)

for sample, sample_df in df.groupby("sample"):
    sample_df = sample_df.sort_values("measurement")
    group = sample_df["group"].iloc[0]
    xs = [
        measurement_pos[m] + group_offset[group]
        for m in sample_df["measurement"]
    ]
    ax.plot(
        xs,
        sample_df["exp-AR"].to_numpy(),
        color="gray",
        alpha=0.6,
        marker="o",
        markersize=4,
        zorder=3,
    )

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:n_hue], labels[:n_hue], title="")
ax.set_xlabel("Measurement")
ax.set_ylabel("amplitude difference from air")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
