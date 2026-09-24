"""
interior_signal_ap.py
=====================
Quantify fluorescence signal inside an eroded wholemount chick embryo
mask and plot mean ± SEM intensity projected onto the normalised
anterior-posterior (A-P) axis [0 = anterior, 1 = posterior].

Pixels are taken from the area remaining after inward erosion of the
mask (rather than a thin ring around the border).

Usage
-----
    python hf_plot_interior_intensity.py

Configuration
-------------
Edit the variables in the CONFIG section below.
"""

import csv
import re
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, uniform_filter1d
from utils import bin_profile

# ─────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────
IMAGE_DIR       = "D:\\Ruoheng_Li\\20260831-dispbead+ROCKibead+WT_staining\\singlechannel\\" # folder containing your images
ANTERIOR_IS_TOP = True         # True  → y=0 is anterior (top of image)
                               # False → y=0 is posterior (flip the axis)
pattern = re.compile(r'([a-z]+)-([a-z]+)([0-9]+)h-([a-z]+[0-9]+)_max\.tif')
ECM='lam'
N_BINS          = 100           # number of A-P bins
EROSION_PX      = 40           # inward erosion depth (px) of the sampling region
SMOOTH_WINDOW   = 5            # light smoothing of the final curve (bins); set 1 to disable
# ─────────────────────────────────────────────


def load_nodes(path: str) -> dict[str, tuple[float, float]]:
    """Read node.csv → {sample_name: (X, Y)}."""
    nodes: dict[str, tuple[float, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            nodes[row["name"]] = (float(row["X"]), float(row["Y"]))
    return nodes


def mask_ap_bounds(mask: np.ndarray) -> tuple[float, float]:
    ys = np.where(mask)[0]
    return float(ys.min()), float(ys.max())


def y_to_ap(y: float, y_min: float, y_max: float, anterior_top: bool) -> float:
    ap = (y - y_min) / (y_max - y_min)
    if not anterior_top:
        ap = 1.0 - ap
    return ap


def extract_eroded_interior(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """
    Return a boolean mask of the area remaining after inward erosion.
    """
    struct = np.ones((3, 3), dtype=bool)
    return binary_erosion(mask, structure=struct, iterations=erosion_px)


def sample_interior_intensity(
    img: np.ndarray,
    interior: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
    anterior_top: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each pixel inside the eroded mask, record its normalised A-P
    position and the raw fluorescence value.

    Returns (ap_positions, intensities) as 1-D arrays.
    """
    ys, xs = np.where(interior)

    y_min, y_max = mask_ap_bounds(mask)
    ap = y_to_ap(ys.astype(np.float64), y_min, y_max, anterior_top)

    intensities = img[ys, xs].astype(np.float64)
    return ap, intensities


def group_mean_sem(
    values: list[np.ndarray],
    normalize: bool = True,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean ± SEM across profiles in one treatment–time group."""
    stack = np.vstack(values)
    if normalize:
        row_max = np.nanmax(stack, axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        stack_norm = stack / row_max
    else:
        stack_norm = stack

    mean = np.nanmean(stack_norm, axis=0)
    sem = np.nanstd(stack_norm, axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(stack_norm), axis=0)
    )
    if smooth_window != 1:
        mean = nan_uniform_filter1d(mean, size=smooth_window)
        sem = nan_uniform_filter1d(sem, size=smooth_window)
    return mean, sem


def style_ap_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(
        "Normalised A-P position  (0 = anterior, 1 = posterior)", fontsize=10,
    )
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def nan_uniform_filter1d(values: np.ndarray, size: int) -> np.ndarray:
    """Moving average that ignores NaNs.

    ``scipy.ndimage.uniform_filter1d`` uses a running sum, so a single NaN
    (empty A-P bin) contaminates the entire curve.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.isfinite(values).astype(np.float64)
    filled = np.where(weights, values, 0.0)
    num = uniform_filter1d(filled, size=size, mode="nearest")
    den = uniform_filter1d(weights, size=size, mode="nearest")
    out = np.full_like(values, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def smooth_profile(profile: np.ndarray) -> np.ndarray:
    if SMOOTH_WINDOW == 1:
        return profile
    return nan_uniform_filter1d(profile, size=SMOOTH_WINDOW)


profiles_by_group: dict[tuple[str, str], list[tuple[str, np.ndarray, float | None]]] = defaultdict(list)
centres_ref: np.ndarray | None = None
nodes = load_nodes(os.path.join(IMAGE_DIR, "node.csv"))

for filename in os.listdir(IMAGE_DIR):
    match = pattern.match(filename)
    if match:
        ECMname = match.group(1)
        treatment = match.group(2)
        time = match.group(3)
        sample = match.group(4)
        samplename = treatment + time + "h-" + sample
        if ECMname != ECM:
            continue
        print(f"Processing {samplename}")
        maskname = "mask-" + samplename + "_max.tif"
        img = cv2.imread(IMAGE_DIR + filename, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(IMAGE_DIR + maskname, cv2.IMREAD_GRAYSCALE)
        interior = extract_eroded_interior(mask, EROSION_PX)
        if not np.any(interior):
            print(f"  skip {samplename}: eroded mask is empty")
            continue
        ap, intensities = sample_interior_intensity(
            img, interior, mask, N_BINS, ANTERIOR_IS_TOP
        )

        centres, means, _ = bin_profile(ap, intensities, N_BINS)

        node_ap: float | None = None
        if samplename in nodes:
            _, node_y = nodes[samplename]
            y_min, y_max = mask_ap_bounds(mask)
            node_ap = y_to_ap(node_y, y_min, y_max, ANTERIOR_IS_TOP)
            print(f"  node AP = {node_ap:.3f}")
        else:
            print(f"  no node for {samplename}")

        profiles_by_group[(treatment, time)].append((sample, means, node_ap))
        if centres_ref is None:
            centres_ref = centres

treatments = sorted({t for t, _ in profiles_by_group})
times = sorted({tm for _, tm in profiles_by_group}, key=int)
color_map = {t: plt.cm.tab10(i) for i, t in enumerate(treatments)}

fig, axs = plt.subplots(1, len(times), figsize=(5 * len(times), 3.5))
if len(times) == 1:
    axs = [axs]

for ax, time in zip(axs, times):
    for treatment in treatments:
        entries = profiles_by_group.get((treatment, time))
        if not entries:
            continue
        profiles = [p for _, p, _ in entries]
        mean, sem = group_mean_sem(profiles, normalize=False, smooth_window=SMOOTH_WINDOW)
        color = color_map[treatment]
        ax.fill_between(
            centres_ref, mean - sem, mean + sem,
            color=color, alpha=0.25, zorder=1,
        )
        ax.plot(
            centres_ref, mean, color=color, lw=2.0, zorder=2,
            label=f"{treatment}  (n = {len(profiles)})",
        )

    ax.set_title(f"{time} h", fontsize=10)
    style_ap_axis(ax)

axs[0].set_ylabel("Interior fluorescence", fontsize=10)

fig.tight_layout()
fig.savefig(IMAGE_DIR + ECM + "_interior_signal.png", dpi=300, bbox_inches="tight")

fig_ind, axs_ind = plt.subplots(1, len(times), figsize=(5 * len(times), 3.5))
if len(times) == 1:
    axs_ind = [axs_ind]

for ax, time in zip(axs_ind, times):
    n_embryos = sum(
        len(profiles_by_group.get((treatment, time), []))
        for treatment in treatments
    )
    colors = plt.cm.tab20(np.arange(max(n_embryos, 1)) % 20)
    embryo_i = 0
    for treatment in treatments:
        entries = profiles_by_group.get((treatment, time))
        if not entries:
            continue
        for sample, profile, node_ap in entries:
            color = colors[embryo_i]
            ax.plot(
                centres_ref,
                smooth_profile(profile),
                color=color,
                lw=1.4,
                label=f"{treatment}-{sample}",
            )
            if node_ap is not None:
                ax.axvline(node_ap, color=color, lw=1.2, ls="--", zorder=3)
            embryo_i += 1

    ax.set_title(f"{time} h", fontsize=10)
    style_ap_axis(ax)

axs_ind[0].set_ylabel("Interior fluorescence", fontsize=10)

fig_ind.tight_layout()
fig_ind.savefig(
    IMAGE_DIR + ECM + "_indivitual_interior_signal.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
