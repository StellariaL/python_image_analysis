"""
Measure mean intensity in 9 ROIs per embryo for fn and lam channels.

CSV has one row per ROI. Boxplots use the mean of the 3 ROIs at each
location, with location on the x-axis and treatment as hue. Points from
the same embryo are connected across locations with gray lines.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from roifile import roiread

# ─────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────
IMAGE_DIR = r"D:\Ruoheng_Li\20260831-dispbead+ROCKibead+WT_staining\dispbead\singlechannel"
CHANNELS = ["fn", "lam"]
ROI_ZIPS = ["bead_rois.zip", "a_p_rois.zip"]
CSV_PATH = os.path.join(IMAGE_DIR, "dispbead-measurements.csv")

LOCATION_BY_ROI = {
    1: "bead",
    2: "bead",
    3: "bead",
    4: "a_blastoderm",
    5: "a_blastoderm",
    6: "a_blastoderm",
    7: "posterior",
    8: "posterior",
    9: "posterior",
}
LOCATION_ORDER = ["bead", "a_blastoderm", "posterior"]
TREATMENT_ORDER = ["ctrl", "disp"]
TREATMENT_COLORS = {
    "ctrl": "#1f77b4",
    "disp": "#d62728",
}
BOX_SPAN = 0.8

ROI_NAME_PATTERN = re.compile(r"^([a-z]+)(\d+h)-([a-z]+\d+)-(\d+)$")
# ─────────────────────────────────────────────


def parse_roi_name(name: str) -> tuple[str, str, int]:
    match = ROI_NAME_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unrecognised ROI name: {name}")
    treatment, _time, embryo, roi_number = match.groups()
    return treatment, embryo, int(roi_number)


def load_rois(image_dir: str, zip_names: list[str]) -> list:
    rois = []
    for zip_name in zip_names:
        rois.extend(roiread(os.path.join(image_dir, zip_name)))
    return rois


def roi_mean_intensity(img: np.ndarray, roi) -> float:
    height, width = img.shape[:2]
    y1 = max(0, int(roi.top))
    y2 = min(height, int(roi.bottom))
    x1 = max(0, int(roi.left))
    x2 = min(width, int(roi.right))
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return float("nan")
    return float(np.mean(patch, dtype=np.float64))


def measure_intensities() -> pd.DataFrame:
    rois = load_rois(IMAGE_DIR, ROI_ZIPS)
    rows = []
    for channel in CHANNELS:
        image_cache: dict[str, np.ndarray] = {}
        for roi in rois:
            treatment, embryo, roi_number = parse_roi_name(roi.name)
            image_stem = f"{channel}-{treatment}4h-{embryo}_max.tif"
            image_path = os.path.join(IMAGE_DIR, image_stem)
            if image_stem not in image_cache:
                image_cache[image_stem] = tifffile.imread(image_path)
            intensity = roi_mean_intensity(image_cache[image_stem], roi)
            rows.append(
                {
                    "embryo": embryo,
                    "treatment": treatment,
                    "channel": channel,
                    "roi_number": roi_number,
                    "location": LOCATION_BY_ROI[roi_number],
                    "intensity": intensity,
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["channel", "treatment", "embryo", "roi_number"],
        ignore_index=True,
    )
    return df


def hue_offsets(hue_order: list[str], span: float = BOX_SPAN) -> dict[str, float]:
    n_hue = len(hue_order)
    dodge = span / n_hue
    return {
        treatment: (i - (n_hue - 1) / 2) * dodge
        for i, treatment in enumerate(hue_order)
    }


def plot_channel(ax, data: pd.DataFrame, channel: str) -> None:
    plot_data = (
        data.loc[data["channel"] == channel]
        .groupby(["embryo", "treatment", "location"], as_index=False)["intensity"]
        .mean()
    )
    location_pos = {loc: i for i, loc in enumerate(LOCATION_ORDER)}
    offsets = hue_offsets(TREATMENT_ORDER)

    sns.boxplot(
        x="location",
        y="intensity",
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

    for (_embryo, treatment), embryo_df in plot_data.groupby(["embryo", "treatment"]):
        embryo_df = embryo_df.dropna(subset=["intensity"]).sort_values(
            "location",
            key=lambda s: s.map(location_pos),
        )
        if len(embryo_df) < 2:
            continue
        xs = [
            location_pos[loc] + offsets[treatment]
            for loc in embryo_df["location"]
        ]
        ax.plot(
            xs,
            embryo_df["intensity"].to_numpy(),
            color="gray",
            alpha=0.6,
            zorder=2,
        )

    sns.stripplot(
        x="location",
        y="intensity",
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
    ax.set_ylabel("mean intensity")
    ax.set_title(channel)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    df = measure_intensities()
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {CSV_PATH} ({len(df)} rows)")

    sns.set_theme(style="white")
    for channel in CHANNELS:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_channel(ax, df, channel)
        fig.tight_layout()
        out_path = os.path.join(IMAGE_DIR, f"dispbead-measurements_{channel}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
