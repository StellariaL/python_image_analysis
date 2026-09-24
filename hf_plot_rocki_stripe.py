"""
Z-stack intensity profiles from a 150 px vertical stripe.

For each Series/channel Z-stack in 10x-ROCKi:
  1. Crop a 150 px-wide stripe centered on coords.csv x (full image height)
  2. Sum intensity over z, then average over x → 1D profile along y
  3. Shift y by subtracting coords.csv y
  4. Plot: 2 channels × 2 times, each with ctrl and ROCKi replicates
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMAGE_DIR = r"D:\Ruoheng_Li\20260817\10x-ROCKi"
COORDS_CSV = os.path.join(IMAGE_DIR, "coords.csv")
OUTPUT_PNG = os.path.join(IMAGE_DIR, "zstack_stripe_profiles.png")
STRIPE_WIDTH = 150
CHANNELS = ("C1", "C2")
TIMES = ("4h", "6h")
TREATMENTS = ("ctrl", "ROCKi")
TREATMENT_COLORS = {"ctrl": "#1f77b4", "ROCKi": "#d62728"}
FILENAME_RE = re.compile(
    r"10x-ROCKibead-c156234e156234_Series(\d+)_C(\d)_ZStack\.tif$"
)


def stripe_profile(path: str, x_center: float, stripe_width: int) -> np.ndarray:
    """Sum over z and average over a vertical stripe centered at x_center."""
    stack = tifffile.memmap(path)
    if stack.ndim != 3:
        raise ValueError(f"{path}: expected 3D ZYX stack, got {stack.shape}")

    _, height, width = stack.shape
    x0 = int(round(x_center)) - stripe_width // 2
    x1 = x0 + stripe_width
    if x0 < 0 or x1 > width:
        print(
            f"  Warning: stripe [{x0}:{x1}] clipped to image width {width} in "
            f"{os.path.basename(path)}"
        )
        x0 = max(0, x0)
        x1 = min(width, x1)
        if x1 <= x0:
            raise ValueError(f"{path}: stripe empty after clipping at x={x_center}")

    stripe = np.asarray(stack[:, :, x0:x1])
    summed_z = np.sum(stripe, axis=0, dtype=np.float64)
    return np.mean(summed_z, axis=1)


def load_zstack_paths(image_dir: str) -> dict[tuple[int, str], str]:
    paths: dict[tuple[int, str], str] = {}
    for name in os.listdir(image_dir):
        match = FILENAME_RE.match(name)
        if not match:
            continue
        series = int(match.group(1))
        channel = f"C{match.group(2)}"
        paths[(series, channel)] = os.path.join(image_dir, name)
    return paths


def main() -> None:
    coords = pd.read_csv(COORDS_CSV)
    paths = load_zstack_paths(IMAGE_DIR)
    expected = 12 * len(CHANNELS)
    if len(paths) != expected:
        raise FileNotFoundError(
            f"Expected {expected} Z-stacks, found {len(paths)} in {IMAGE_DIR}"
        )

    profiles: dict[tuple[str, str, str], list[tuple[int, np.ndarray, np.ndarray]]] = {
        (ch, time, treatment): []
        for ch in CHANNELS
        for time in TIMES
        for treatment in TREATMENTS
    }

    for _, row in coords.iterrows():
        series = int(row["no"])
        x_center = float(row["x"])
        y_offset = float(row["y"])
        treatment = str(row["treatment"])
        time = str(row["time"])

        for channel in CHANNELS:
            path = paths.get((series, channel))
            if path is None:
                raise FileNotFoundError(f"Missing Z-stack for Series{series:02d} {channel}")

            print(f"Processing Series{series:02d} {channel}  {treatment} {time}")
            intensity = stripe_profile(path, x_center, STRIPE_WIDTH)
            y_shifted = np.arange(intensity.size, dtype=np.float64) - y_offset
            profiles[(channel, time, treatment)].append((series, y_shifted, intensity))

    fig, axs = plt.subplots(
        len(CHANNELS),
        len(TIMES),
        figsize=(10, 8),
        sharex=True,
        sharey="row",
    )

    for i, channel in enumerate(CHANNELS):
        for j, time in enumerate(TIMES):
            ax = axs[i, j]
            for treatment in TREATMENTS:
                series_list = profiles[(channel, time, treatment)]
                color = TREATMENT_COLORS[treatment]
                for k, (series, y_shifted, intensity) in enumerate(series_list):
                    ax.plot(
                        y_shifted,
                        intensity,
                        color=color,
                        lw=1.2,
                        alpha=0.85,
                        label=treatment if k == 0 else None,
                    )
            ax.axvline(0, color="0.6", lw=0.8, ls="--", zorder=0)
            ax.set_title(f"{channel}  {time}", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            if i == len(CHANNELS) - 1:
                ax.set_xlabel("y (px, shifted)", fontsize=10)
            if j == 0:
                ax.set_ylabel("Z-summed mean intensity", fontsize=10)
            ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
