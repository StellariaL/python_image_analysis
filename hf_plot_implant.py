"""
Plot gel implant ROI outlines from a TIFF stack and manual ROI zip.

For each sample (treatment x well), superimpose ROI outlines from all
timepoints aligned by centroid. Colour by timepoint (0h, 2h, 4h).
Slices without an assigned ROI are treated as a single point.
"""

import re
import zipfile
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roifile
import tifffile

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
STACK_PATH = "20260711-gelimplant-stack.tif"
ROI_ZIP_PATH = "manual_roi.zip"
OUTPUT_PATH = "gelimplant_roi_overlay.png"

TREATMENTS = ["1", "02", "05"]
SAMPLES = [f"c{i}" for i in range(1, 5)] + [f"e{i}" for i in range(1, 5)]
TIMEPOINTS = ["0h", "2h", "4h"]
TIMEPOINT_COLORS = {"0h": "#1f77b4", "2h": "#ff7f0e", "4h": "#2ca02c"}

LABEL_PATTERN = re.compile(
    r"260711-gelimplant_(?P<treatment>\d+)(?P<sample>[ce]\d)-(?P<time>\d+h)_ch00\.tif"
)
ROI_INDEX_PATTERN = re.compile(r"^(\d+)-")


def load_slice_labels(stack_path: str) -> list[str]:
    with tifffile.TiffFile(stack_path) as tif:
        return list(tif.imagej_metadata["Labels"])


def load_rois_by_slice(roi_zip_path: str) -> dict[int, np.ndarray]:
    rois_by_slice: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(roi_zip_path) as zf:
        for name in zf.namelist():
            match = ROI_INDEX_PATTERN.match(name)
            if not match:
                continue
            slice_idx = int(match.group(1)) - 1
            roi = roifile.ImagejRoi.frombytes(zf.read(name))
            rois_by_slice[slice_idx] = roi.coordinates()
    return rois_by_slice


def parse_label_groups(labels: list[str]) -> dict[tuple[str, str], dict[str, int]]:
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for slice_idx, label in enumerate(labels):
        match = LABEL_PATTERN.match(label)
        if not match:
            continue
        key = (match.group("treatment"), match.group("sample"))
        groups[key][match.group("time")] = slice_idx
    return groups


def align_by_centroid(coords: np.ndarray) -> np.ndarray:
    return coords - coords.mean(axis=0)


def get_roi_coords(
    slice_idx: int,
    rois_by_slice: dict[int, np.ndarray],
) -> tuple[np.ndarray, bool]:
    if slice_idx in rois_by_slice:
        return rois_by_slice[slice_idx].astype(float), False
    return np.array([[0.0, 0.0]]), True


def plot_sample_ax(
    ax: plt.Axes,
    treatment: str,
    sample: str,
    timepoints: dict[str, int],
    rois_by_slice: dict[int, np.ndarray],
) -> None:
    for timepoint in TIMEPOINTS:
        if timepoint not in timepoints:
            continue

        coords, is_point = get_roi_coords(timepoints[timepoint], rois_by_slice)
        aligned = align_by_centroid(coords)
        color = TIMEPOINT_COLORS[timepoint]

        if is_point:
            ax.plot(
                aligned[0, 0],
                aligned[0, 1],
                "o",
                color=color,
                markersize=4,
                label=timepoint,
            )
        else:
            closed = np.vstack([aligned, aligned[:1]])
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                color=color,
                linewidth=1.5,
                label=timepoint,
            )

    ax.set_title(f"{treatment}-{sample}")
    ax.set_aspect("equal")
    ax.axhline(0, color="0.85", linewidth=0.5)
    ax.axvline(0, color="0.85", linewidth=0.5)
    ax.tick_params(labelsize=7)


def main() -> None:
    labels = load_slice_labels(STACK_PATH)
    rois_by_slice = load_rois_by_slice(ROI_ZIP_PATH)
    groups = parse_label_groups(labels)

    sample_keys = [
        (treatment, sample)
        for treatment in TREATMENTS
        for sample in SAMPLES
        if (treatment, sample) in groups
    ]

    n_cols = 4
    n_rows = int(np.ceil(len(sample_keys) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    for ax, (treatment, sample) in zip(axes.ravel(), sample_keys):
        plot_sample_ax(ax, treatment, sample, groups[(treatment, sample)], rois_by_slice)

    for ax in axes.ravel()[len(sample_keys) :]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=TIMEPOINT_COLORS[tp], linewidth=2, label=tp)
        for tp in TIMEPOINTS
    ]
    fig.legend(handles=handles, loc="upper right", title="Timepoint")
    fig.supxlabel("x offset from centroid (px)")
    fig.supylabel("y offset from centroid (px)")
    fig.suptitle("Gel implant ROI outlines aligned by centroid", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
