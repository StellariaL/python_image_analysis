'''
Estimate aligned_time from l_notochord using a cached LOWESS fit,
then compute time remaining until mean fold initiation.

Usage:
python hf_estimate_time.py data/260821-init_nc.csv
python hf_estimate_time.py file1.csv file2.csv
python hf_estimate_time.py data/260821-init_nc.csv --output data/out.csv
python hf_estimate_time.py data/260821-init_nc.csv --refit
python hf_estimate_time.py data/260821-init_nc.csv --scale 0.62

cache file: data/l_notochord_lowess.csv
'''

import argparse
import os

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

NOTOCHORD_CSV = os.path.join("data", "l_notochord.csv")
OFFSET_CSV = os.path.join("data", "offset.csv")
FIT_CSV = os.path.join("data", "l_notochord_lowess.csv")

LOESS_FRAC = 0.4
N_GRID = 500


def mean_aligned_init(offset_path: str = OFFSET_CSV) -> float:
    offset = pd.read_csv(offset_path)
    return float(offset["aligned_init"].mean())


def build_lowess_fit(
    notochord_path: str = NOTOCHORD_CSV,
    frac: float = LOESS_FRAC,
    n_grid: int = N_GRID,
) -> pd.DataFrame:
    df = pd.read_csv(notochord_path)
    x = df["aligned_time"].to_numpy(dtype=float)
    y = df["l_notochord"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    time_grid = np.linspace(x.min(), x.max(), n_grid)
    length_fit = lowess(y, x, frac=frac, xvals=time_grid, return_sorted=False)
    return pd.DataFrame({"aligned_time": time_grid, "l_notochord": length_fit})


def save_fit(
    fit: pd.DataFrame,
    mean_init: float,
    path: str = FIT_CSV,
    frac: float = LOESS_FRAC,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write(f"# mean_aligned_init={mean_init}\n")
        handle.write(f"# frac={frac}\n")
        fit.to_csv(handle, index=False)


def load_fit(path: str = FIT_CSV) -> tuple[pd.DataFrame, float, float]:
    meta: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            key, value = line[1:].strip().split("=", 1)
            meta[key.strip()] = value.strip()

    fit = pd.read_csv(path, comment="#")
    mean_init = float(meta["mean_aligned_init"])
    frac = float(meta.get("frac", LOESS_FRAC))
    return fit, mean_init, frac


def get_or_build_fit(
    fit_path: str = FIT_CSV,
    notochord_path: str = NOTOCHORD_CSV,
    offset_path: str = OFFSET_CSV,
    frac: float = LOESS_FRAC,
    refit: bool = False,
) -> tuple[pd.DataFrame, float]:
    if os.path.exists(fit_path) and not refit:
        fit, mean_init, _ = load_fit(fit_path)
        return fit, mean_init

    mean_init = mean_aligned_init(offset_path)
    fit = build_lowess_fit(notochord_path, frac=frac)
    save_fit(fit, mean_init, path=fit_path, frac=frac)
    return fit, mean_init


def invert_length_to_time(
    lengths: np.ndarray,
    fit_time: np.ndarray,
    fit_length: np.ndarray,
) -> np.ndarray:
    """Map notochord length onto the LOWESS curve to recover aligned_time."""
    order = np.argsort(fit_length)
    length_sorted = fit_length[order]
    time_sorted = fit_time[order]

    unique_length, unique_idx = np.unique(length_sorted, return_index=True)
    unique_time = time_sorted[unique_idx]

    estimated = np.full(lengths.shape, np.nan, dtype=float)
    valid = np.isfinite(lengths)
    if not valid.any():
        return estimated

    values = lengths[valid]
    estimated[valid] = np.interp(values, unique_length, unique_time)

    below = valid & (lengths < unique_length[0])
    above = valid & (lengths > unique_length[-1])
    if below.any():
        slope = (unique_time[1] - unique_time[0]) / (unique_length[1] - unique_length[0])
        estimated[below] = unique_time[0] + (lengths[below] - unique_length[0]) * slope
    if above.any():
        slope = (unique_time[-1] - unique_time[-2]) / (
            unique_length[-1] - unique_length[-2]
        )
        estimated[above] = unique_time[-1] + (lengths[above] - unique_length[-1]) * slope
    return estimated


def annotate_csv(
    csv_path: str,
    fit: pd.DataFrame,
    mean_init: float,
    output_path: str | None = None,
    scale: float = 1.0,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "l_notochord" not in df.columns:
        raise KeyError(f"{csv_path} has no 'l_notochord' column")

    lengths = df["l_notochord"].to_numpy(dtype=float) * scale
    if scale != 1.0:
        df["l_notochord_scaled"] = lengths
    estimated = invert_length_to_time(
        lengths,
        fit["aligned_time"].to_numpy(dtype=float),
        fit["l_notochord"].to_numpy(dtype=float),
    )
    df["estimated_time"] = estimated
    df["time_to_fold_init"] = mean_init - estimated

    dest = output_path or csv_path
    df.to_csv(dest, index=False)

    fit_min = float(fit["l_notochord"].min())
    fit_max = float(fit["l_notochord"].max())
    outside = np.isfinite(lengths) & ((lengths < fit_min) | (lengths > fit_max))
    if outside.any():
        print(
            f"Warning: {int(outside.sum())} length(s) in {csv_path} are outside "
            f"the fitted range [{fit_min:.1f}, {fit_max:.1f}]; times were extrapolated."
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate aligned_time from l_notochord using a cached LOWESS fit, "
            "then compute time remaining until mean fold initiation."
        ),
    )
    parser.add_argument(
        "csv",
        nargs="+",
        help="CSV file(s) with an l_notochord column, e.g. data/260821-init_nc.csv",
    )
    parser.add_argument(
        "--output",
        help="Output CSV (only with a single input; default: overwrite each input)",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Rebuild the LOWESS calibration curve instead of using the cache",
    )
    parser.add_argument(
        "--fit",
        default=FIT_CSV,
        help=f"Path to cached LOWESS fit CSV (default: {FIT_CSV})",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "Multiply l_notochord by this factor before estimating time, e.g. to "
            "convert pixels to the calibration curve's units (default: 1.0)"
        ),
    )
    args = parser.parse_args()

    if args.output and len(args.csv) > 1:
        parser.error("--output can only be used with a single input CSV")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be a positive number")

    fit, mean_init = get_or_build_fit(fit_path=args.fit, refit=args.refit)
    print(f"Using LOWESS fit from {args.fit}")
    print(f"Mean aligned_init (fold initiation): {mean_init}")
    if args.scale != 1.0:
        print(f"Scaling l_notochord by {args.scale}")

    for csv_path in args.csv:
        output_path = args.output if len(args.csv) == 1 else None
        result = annotate_csv(
            csv_path, fit, mean_init, output_path=output_path, scale=args.scale
        )
        dest = output_path or csv_path
        print(f"Wrote estimated_time and time_to_fold_init for {len(result)} rows to {dest}")


if __name__ == "__main__":
    main()
