'''
reusable functions
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def bin_profile(
    axis: np.ndarray,
    values: np.ndarray,
    n_bins: int,
    limits: tuple[float, float]=(0,1)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin (axis, values) into n_bins equal-width bins over [0, 1].

    Returns (bin_centres, mean_intensity, sem_intensity).
    """
    edges = np.linspace(limits[0], limits[1], n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    sems  = np.full(n_bins, np.nan)

    for i in range(n_bins):
        sel = (axis >= edges[i]) & (axis < edges[i + 1])
        if sel.sum() > 1:
            vals = values[sel]
            means[i] = vals.mean()
            sems[i]  = vals.std(ddof=1) / np.sqrt(sel.sum())

    return centres, means, sems


def plot_profiles(
    x: np.ndarray,
    values: list[np.ndarray],
    ax=None,
    normalize: bool=True,
    smooth_window: int=5, # set 1 to disable
) -> None:
    """
    For multiple value profiles along same x:
    normalize each to its own maximum (set normalize=False to disable),
    compute grand mean ± SEM across all,
    then plot:
        grand mean as dark line; 
        SEM as lighter band;
        individual profiles as translucent lines in background
    """
    lines=[]

    stack = np.vstack(values)          # shape (n_images, n_bins)

    # Normalise each image's profile to its own maximum before averaging
    # so brightness differences between images do not dominate.
    if normalize:
        row_max = np.nanmax(stack, axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        stack_norm = stack / row_max
    else:
        stack_norm=stack

    grand_mean = np.nanmean(stack_norm, axis=0)
    grand_sem  = np.nanstd(stack_norm, axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(stack_norm), axis=0)
    )
    if smooth_window!=1:
        grand_mean=uniform_filter1d(grand_mean,size=smooth_window,mode="nearest")
        grand_sem  = uniform_filter1d(grand_sem,size=smooth_window,mode="nearest")
    # ── figure ──
    if ax==None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    

    # Individual traces (faint)
    for row in stack_norm:
        lines.append(ax.plot(x, row, color="#4a9eca", alpha=0.25, lw=0.8, zorder=1))

    # SEM band
    sem_band=ax.fill_between(
        x,
        grand_mean - grand_sem,
        grand_mean + grand_sem,
        color="#1a6fa0",
        alpha=0.25,
        zorder=2,
        label="_nolegend_",
    )

    # Grand mean
    gm_line=ax.plot(x, grand_mean, color="#1a6fa0", lw=2.0, zorder=3,
            label=f"Mean ± SEM  (n = {len(values)})")

    return ax,gm_line,sem_band,lines