"""
border_signal_ap.py
====================
Quantify fluorescence signal along the border of wholemount chick embryo
confocal images and plot mean ± SEM intensity projected onto the
normalised anterior-posterior (A-P) axis [0 = anterior, 1 = posterior].

Only images matching DATE, EXPERIMENT and TIME are measured.

Images are multichannel; every channel listed in CHANNELS is measured
separately and saved as
[channel label]-[date]-[experiment]-[time]h_border_intensity.png.

Usage
-----
    python border_signal_ap.py

Configuration
-------------
Edit the variables in the CONFIG section below.
"""

import re
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion, uniform_filter1d
from utils import bin_profile

# ─────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────
IMAGE_DIR       = "D:\\Ruoheng_Li\\20260925\\4x\\rotated\\" # folder containing your images
DATE            = "0911"     # only images from this date are measured
EXPERIMENT      = "dispbeadant"   # only images from this experiment are measured
TIME            = "2"          # only images from this time point (hours) are measured
pattern = re.compile(r'([0-9]+)-([a-z]+)-([a-z]+)([0-9]+)h-([0-9+])x-([a-z]+)-([a-z]+[0-9]+)_max\.tif')
# example: 260821-dispbead-disp1h-4x-v-e1_max.tif
CHANNELS        = {4: "fn"}   # channel number → label; index starts from 1
N_BINS          = 100           # number of A-P bins
BORDER_BAND_PX  = 40           # inward erosion depth (px) used to sample signal
SMOOTH_WINDOW   = 5            # light smoothing of the final curve (bins); set 1 to disable
# ─────────────────────────────────────────────


def load_channel_stack(path: str) -> np.ndarray:
    """
    Read a multichannel 2-D image and return it as (channel, y, x).
    """
    img = tifffile.imread(path)
    if img.ndim != 3:
        raise ValueError(
            f"{os.path.basename(path)}: expected a multichannel 2-D image, got shape {img.shape}"
        )
    if img.shape[-1] <= img.shape[0]:          # channel-last (y, x, channel)
        img = np.moveaxis(img, -1, 0)
    return img


def extract_border_ring(mask: np.ndarray, band_px: int) -> np.ndarray:
    """
    Return a boolean mask of a thin annular ring just inside the embryo border.
    Computed as: original_mask XOR eroded_mask (eroded by band_px iterations).
    """
    struct = np.ones((3, 3), dtype=bool)
    dilated=binary_dilation(mask,structure=struct, iterations=band_px)
    eroded = binary_erosion(mask, structure=struct, iterations=band_px)
    return dilated & ~eroded


def sample_border_intensity(
    img: np.ndarray,
    ring: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each pixel in the border ring, record its normalised A-P position
    and the raw fluorescence value.

    Returns (ap_positions, intensities) as 1-D arrays.
    """
    ys, xs = np.where(ring)

    # Normalise y to [0, 1] within the embryo bounding box; y=0 is anterior
    y_min, y_max = np.where(mask)[0].min(), np.where(mask)[0].max()
    ap = (ys - y_min) / (y_max - y_min)

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
        mean = uniform_filter1d(mean, size=smooth_window, mode="nearest")
        sem = uniform_filter1d(sem, size=smooth_window, mode="nearest")
    return mean, sem


def style_ap_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(
        "Normalised A-P position  (0 = anterior, 1 = posterior)", fontsize=10,
    )
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def output_path(label: str, suffix: str) -> str:
    return os.path.join(
        IMAGE_DIR, f"{label}-{DATE}-{EXPERIMENT}-{TIME}h{suffix}.png"
    )


def smooth_profile(profile: np.ndarray) -> np.ndarray:
    if SMOOTH_WINDOW == 1:
        return profile
    return uniform_filter1d(profile, size=SMOOTH_WINDOW, mode="nearest")


def plot_group_profiles(
    profiles_by_group: dict[tuple[str, str], list[tuple[str, np.ndarray]]],
    centres: np.ndarray,
    label: str,
) -> None:
    """Mean ± SEM per treatment, one panel per time point."""
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
            profiles = [p for _, p in entries]
            mean, sem = group_mean_sem(profiles, normalize=False, smooth_window=SMOOTH_WINDOW)
            color = color_map[treatment]
            ax.fill_between(
                centres, mean - sem, mean + sem,
                color=color, alpha=0.25, zorder=1,
            )
            ax.plot(
                centres, mean, color=color, lw=2.0, zorder=2,
                label=f"{treatment}  (n = {len(profiles)})",
            )

        ax.set_title(f"{time} h", fontsize=10)
        style_ap_axis(ax)

    axs[0].set_ylabel(f"{label} border fluorescence", fontsize=10)

    fig.tight_layout()
    fig.savefig(
        output_path(label, "_border_intensity"), dpi=300, bbox_inches="tight"
    )


def plot_individual_profiles(
    profiles_by_group: dict[tuple[str, str], list[tuple[str, np.ndarray]]],
    centres: np.ndarray,
    label: str,
) -> None:
    """One curve per embryo, one panel per time point."""
    treatments = sorted({t for t, _ in profiles_by_group})
    times = sorted({tm for _, tm in profiles_by_group}, key=int)

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
            for sample, profile in entries:
                ax.plot(
                    centres,
                    smooth_profile(profile),
                    color=colors[embryo_i],
                    lw=1.4,
                    label=f"{treatment}-{sample}",
                )
                embryo_i += 1

        ax.set_title(f"{time} h", fontsize=10)
        style_ap_axis(ax)

    axs_ind[0].set_ylabel(f"{label} border fluorescence", fontsize=10)

    fig_ind.tight_layout()
    fig_ind.savefig(
        output_path(label, "_individual_border_intensity"),
        dpi=300,
        bbox_inches="tight",
    )


profiles_by_channel: dict[str, dict[tuple[str, str], list[tuple[str, np.ndarray]]]] = {
    label: defaultdict(list) for label in CHANNELS.values()
}
centres_ref: np.ndarray | None = None

for filename in os.listdir(IMAGE_DIR):
    match = pattern.match(filename)
    if match:
        date=match.group(1)
        experiment = match.group(2)
        treatment=match.group(3)
        time = match.group(4)
        magnification = match.group(5)
        side = match.group(6)
        sample = match.group(7)
        if (date, experiment, time) != (DATE, EXPERIMENT, TIME):
            continue
        samplename = treatment + time + "h-" + sample
        print(f"Processing {samplename}")
        maskname = "mask-" + filename
        stack = load_channel_stack(IMAGE_DIR + filename)
        mask = cv2.imread(IMAGE_DIR + maskname, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  skip {samplename}: no mask {maskname}")
            continue
        ring = extract_border_ring(mask, BORDER_BAND_PX)

        for number, label in CHANNELS.items():
            if number > stack.shape[0]:
                raise IndexError(
                    f"{filename}: channel {number} ({label}) requested "
                    f"but image has only {stack.shape[0]} channels"
                )
            ap, intensities = sample_border_intensity(
                stack[number - 1], ring, mask, N_BINS
            )

            centres, means, _ = bin_profile(ap, intensities, N_BINS)

            profiles_by_channel[label][(treatment, time)].append((sample, means))
            if centres_ref is None:
                centres_ref = centres

for label, profiles_by_group in profiles_by_channel.items():
    if not profiles_by_group:
        print(f"No profiles for channel {label}")
        continue
    plot_group_profiles(profiles_by_group, centres_ref, label)
    plot_individual_profiles(profiles_by_group, centres_ref, label)

plt.show()
