'''
This script takes a 3D stack and generates a 2D projection of its surface layer.
Good when the surface is curved.

workflow:
1. read stack
2. apply Gaussian blur to each slice
3. for each pixel, find the first z slice in the blurred stack (from z=0 up to
   the peak) that exceeds a local intensity threshold (peak_percentage × peak)
4. take a local max projection in a z window centered on that threshold slice
5. median-filter the resulting z indices, then sample the stack at those z's

parameters:
    r_blur: Gaussian blur sigma (px); large enough to cover characteristic features
    peak_percentage: fraction of per-pixel peak intensity used as threshold
    half_window: half-width of z window for local max projection
    smooth_ker: median-filter kernel size applied to the chosen z indices

outputs:
    *-surf.tif: 2D surface projection; same XY size as input stack
    *-record.tif: 3D mask (16-bit), projected pixels marked 4096; same XYZ as input

tips:
Open the -record mask in Fiji and merge it to the input stack to visualize
which pixels are taken.
'''

import os
import re

import cv2
import numpy as np
import tifffile as tiff
from scipy.ndimage import median_filter

folder = "E:\\PhD_large_images\\20260512-cellshape\\"
r_blur = 200
half_window = 0
peak_percentage = 0.95
smooth_ker = 200

zrange = 2 * half_window + 1
g_blur_k = 2 * r_blur + 1
record_value = 4096

pattern = re.compile(r'0603-e[0-9]-lamA-[0-9]\.tif')
# pattern = re.compile(r'0603-e2-actin-1\.tif')


def find_threshold_z(blurred: np.ndarray, peak_percentage: float) -> np.ndarray:
    """First z at or above peak_percentage × local peak, searching z=0 .. peak_z."""
    z, _, _ = blurred.shape
    peak_z = np.argmax(blurred, axis=0)
    peak_intensity = np.take_along_axis(
        blurred, peak_z[np.newaxis, :, :], axis=0
    )[0]
    local_threshold = peak_intensity * peak_percentage

    z_indices = np.arange(z, dtype=np.int32)[:, np.newaxis, np.newaxis]
    in_search = z_indices <= peak_z[np.newaxis, :, :]
    above = blurred >= local_threshold[np.newaxis, :, :]
    z_candidates = np.where(in_search & above, z_indices, z + 1)
    threshold_z = z_candidates.min(axis=0)

    no_hit = threshold_z > z
    threshold_z[no_hit] = peak_z[no_hit]
    return threshold_z


def local_max_z(
    stack: np.ndarray,
    center_z: np.ndarray,
    half_window: int,
) -> np.ndarray:
    """Max-intensity z in a window centered on center_z for each (y, x)."""
    z, y, x = stack.shape
    zrange = 2 * half_window + 1
    window = (np.arange(zrange) - half_window).reshape(zrange, 1, 1)

    local_z = center_z[np.newaxis, :, :] + window
    local_z = np.clip(local_z, 0, z - 1)

    y_ind = np.broadcast_to(np.arange(y)[:, None], (zrange, y, x))
    x_ind = np.broadcast_to(np.arange(x)[None, :], (zrange, y, x))
    local_intensities = stack[local_z, y_ind, x_ind]

    local_max_offset = np.argmax(local_intensities, axis=0)
    y_mesh, x_mesh = np.meshgrid(np.arange(y), np.arange(x), indexing='ij')
    return local_z[local_max_offset, y_mesh, x_mesh]


for filename in os.listdir(folder):
    match = pattern.match(filename)
    if not match:
        continue

    input_path = os.path.join(folder, filename)
    projection_path = input_path.removesuffix('.tif') + '-surf.tif'
    record_path = input_path.removesuffix('.tif') + '-record.tif'

    stack = tiff.imread(input_path)
    if stack.ndim != 3:
        raise ValueError("Input image must be a grayscale z-stack (Z, Y, X).")

    z, y, x = stack.shape

    blurred = np.empty((z, y, x), dtype=np.float32)
    for zi in range(z):
        blurred[zi] = cv2.GaussianBlur(
            stack[zi],
            (g_blur_k, g_blur_k),
            sigmaX=r_blur,
        )

    threshold_z = find_threshold_z(blurred, peak_percentage)
    max_z = local_max_z(stack, threshold_z, half_window)

    smooth_z = median_filter(max_z.astype(np.float64), size=smooth_ker)
    surf_z_idx = np.clip(np.round(smooth_z).astype(np.int32), 0, z - 1)

    y_mesh, x_mesh = np.meshgrid(np.arange(y), np.arange(x), indexing='ij')
    projection = stack[surf_z_idx, y_mesh, x_mesh]

    final_z = np.zeros((z, y, x), dtype=np.uint16)
    final_z[surf_z_idx, y_mesh, x_mesh] = record_value

    tiff.imwrite(projection_path, projection)
    tiff.imwrite(record_path, final_z)
