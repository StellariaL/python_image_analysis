"""
border_signal_ap.py
====================
Quantify fluorescence signal along the border of wholemount chick embryo
confocal images and plot mean ± SEM intensity projected onto the
normalised anterior-posterior (A-P) axis [0 = anterior, 1 = posterior].

Usage
-----
    python border_signal_ap.py

Configuration
-------------
Edit the variables in the CONFIG section below.
"""

import re
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, uniform_filter1d
from utils import bin_profile,plot_profiles

# ─────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────
IMAGE_DIR       = "E:\\PhD_large_images\\20260512-ECM\\" # folder containing your images
ANTERIOR_IS_TOP = True         # True  → y=0 is anterior (top of image)
                               # False → y=0 is posterior (flip the axis)
pattern = re.compile(r'([0-9]+-e\d+)-4x-max-([a-z]+)\.tif')
ECM='laminin'
N_BINS          = 100           # number of A-P bins
BORDER_BAND_PX  = 40           # inward erosion depth (px) used to sample signal
SMOOTH_WINDOW   = 5            # light smoothing of the final curve (bins); set 1 to disable
# ─────────────────────────────────────────────



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
    anterior_top: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each pixel in the border ring, record its normalised A-P position
    and the raw fluorescence value.

    Returns (ap_positions, intensities) as 1-D arrays.
    """
    ys, xs = np.where(ring)

    # Normalise y to [0, 1] within the embryo bounding box
    y_min, y_max = np.where(mask)[0].min(), np.where(mask)[0].max()
    ap = (ys - y_min) / (y_max - y_min)          # 0 = top of embryo

    if not anterior_top:                           # flip if posterior is top
        ap = 1.0 - ap

    intensities = img[ys, xs].astype(np.float64)
    return ap, intensities



all_means: list[np.ndarray] = []
centres_ref: np.ndarray | None = None

for filename in os.listdir(IMAGE_DIR):
    match = pattern.match(filename)
    if match:
        samplename=match.group(1) #group index starts from 1
        ECMname = match.group(2)
        if ECMname != ECM:
            continue
        print(f"Processing {samplename}")
        maskname=samplename+"-4x-mask.tif"
        img=cv2.imread(IMAGE_DIR+filename,cv2.IMREAD_GRAYSCALE)
        mask=cv2.imread(IMAGE_DIR+maskname,cv2.IMREAD_GRAYSCALE)
        ring=extract_border_ring(mask,BORDER_BAND_PX)
        ap, intensities = sample_border_intensity(
            img, ring, mask, N_BINS, ANTERIOR_IS_TOP
        )

        centres, means, _ = bin_profile(ap, intensities, N_BINS)

        all_means.append(means)
        if centres_ref is None:
            centres_ref = centres

ax,gm_line,sem_band,indiv_lines=plot_profiles(centres_ref,all_means,
                                              smooth_window=SMOOTH_WINDOW)


ax.set_xlabel("Normalised A-P position  (0 = anterior, 1 = posterior)",
                  fontsize=10)
ax.set_ylabel("Normalised border fluorescence\n(a.u.)", fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

'''
to set line appearances:
gm_line.set_color('purple')
gm_line.set_linewidth(2.5)

sem_band.set_color('orchid')
sem_band.set_alpha(0.15)

plt.setp(indiv_lines, color='purple', linewidth=3, linestyle='--')
'''

plt.tight_layout()
plt.savefig(IMAGE_DIR+ECM+"_border_signal.png", dpi=300, bbox_inches="tight")
plt.show()