"""
Plot averaged AR/Ac values from combined_fit_results.csv.

Each sample (control: c*, treated: e*) is identified by date and sample name.
Air measurements and samples with '6' in notes are excluded. For each sample,
exp-AR is averaged across measurements 1 and 2; if measurement 2 is missing,
measurement 1 alone is used.
Individual points are shown with boxplots grouped by control vs treated.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

CSV_PATH = "combined_fit_results.csv"
OUTPUT_PATH = "combined-summary-diff-avg.png"


def significance_text(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def average_exp_ar(sample_df):
    m1 = sample_df.loc[sample_df["measurement"] == 1, "exp-AR"]
    m2 = sample_df.loc[sample_df["measurement"] == 2, "exp-AR"]

    if not m2.empty:
        if m1.empty:
            return None
        return (m1.iloc[0] + m2.iloc[0]) / 2
    if not m1.empty:
        return m1.iloc[0]
    return None


df = pd.read_csv(CSV_PATH)
df = df[df["sample"] != "air"].copy()
df["measurement"] = df["measurement"].astype(int)

rows = []
for (date, sample), sample_df in df.groupby(["date", "sample"], sort=False):
    if sample_df["notes"].astype(str).str.contains("6", na=False).any():
        continue
    exp_ar = average_exp_ar(sample_df)
    if exp_ar is None:
        continue
    rows.append(
        {
            "date": date,
            "sample": sample,
            "exp-AR": exp_ar,
            "group": "control" if sample.startswith("c") else "treated",
        }
    )

plot_df = pd.DataFrame(rows)
hue_order = ["control", "treated"]

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(4, 4))

sns.boxplot(
    x="group",
    y="exp-AR",
    data=plot_df,
    order=hue_order,
    linewidth=1.5,
    boxprops=dict(facecolor="white", edgecolor="black"),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    medianprops=dict(color="black"),
    flierprops=dict(markerfacecolor="black", markeredgecolor="black"),
    width=0.4,
    ax=ax,
)

sns.stripplot(
    x="group",
    y="exp-AR",
    data=plot_df,
    order=hue_order,
    color="black",
    alpha=0.7,
    jitter=False,
    ax=ax,
)

ctrl = plot_df.loc[plot_df["group"] == "control", "exp-AR"]
treated = plot_df.loc[plot_df["group"] == "treated", "exp-AR"]
u_stat, p_val = stats.mannwhitneyu(treated, ctrl, alternative="less")
t_stat, t_pval = stats.ttest_ind(treated, ctrl, alternative="less")
print(f"Mann-Whitney U test (treated < control): U = {u_stat:.4f}, p = {p_val:.4g}")
print(f"t-test (treated < control): t = {t_stat:.4f}, p = {t_pval:.4g}")

y_max = plot_df["exp-AR"].max()
y_bracket = y_max * 1.05
ax.plot([0, 1], [y_bracket, y_bracket], lw=1.5, c="black")
ax.text(0.5, y_bracket * 1.02, significance_text(p_val), ha="center", fontsize=12, fontweight="bold")
ax.text(0.5, y_bracket * 1.08, f"p = {p_val:.3g}", ha="center", fontsize=10)
ax.set_ylim(top=y_bracket * 1.15)

ax.set_xlabel("")
ax.set_ylabel("amplitude difference from air")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
