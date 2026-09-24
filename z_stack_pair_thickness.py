'''
Compute per-pixel thickness (zB - zA) from paired binary marker stacks.

For each stack pair [samplename]-A.tif and [samplename]-B.tif, both stacks are
expected to have shape (Z, Y, X), identical XY dimensions, and pixel values 0 or
4096 only. At each (y, x) there is exactly one 4096 voxel along z in A (zA) and
one in B (zB).

outputs:
    [samplename]-thickness.tif: 2D int32 image (Y, X) of zB - zA
'''

import os
import re

import numpy as np
import tifffile as tiff

folder = "E:\\PhD_large_images\\20260512-cellshape\\"
marker_value = 4096

pattern = re.compile(r'(.+)-actin-([0-9])-record\.tif$')
#filename format: date-sample-actin-view-record.tif
#eg: 0603-e1-actin-1-record.tif

def find_marker_z(stack: np.ndarray, marker: int) -> np.ndarray:
    """Z index of marker at each (y, x) for a (Z, Y, X) stack."""
    mask = stack == marker
    z_coords = np.arange(stack.shape[0], dtype=np.int32)[:, np.newaxis, np.newaxis]
    z_idx = np.where(mask, z_coords, -1).max(axis=0)
    return z_idx


for filename in os.listdir(folder):
    match = pattern.match(filename)
    if not match:
        continue

    samplename = match.group(1)
    view=match.group(2)
    path_a = os.path.join(folder, filename)
    path_b = os.path.join(folder, f"{samplename}-lamA-{view}-record.tif")
    if not os.path.isfile(path_b):
        print(f"Skipping {filename}: missing {samplename} laminin stack")
        continue

    stack_a = tiff.imread(path_a)
    stack_b = tiff.imread(path_b)

    if stack_a.ndim != 3 or stack_b.ndim != 3:
        raise ValueError(f"{samplename}: both stacks must be 3D (Z, Y, X).")
    if stack_a.shape != stack_b.shape:
        raise ValueError(
            f"{samplename}: shape mismatch A{stack_a.shape} vs B{stack_b.shape}."
        )

    z_a = find_marker_z(stack_a, marker_value)
    z_b = find_marker_z(stack_b, marker_value)

    missing = (z_a < 0) | (z_b < 0)
    if np.any(missing):
        n_missing = int(missing.sum())
        print(f"Warning: {samplename} has {n_missing} XY positions without {marker_value}.")

    thickness = (z_b - z_a).astype(np.int32)
    thickness[missing] = np.iinfo(np.int32).min

    out_path = os.path.join(folder, f"{samplename}-{view}-thickness.tif")
    tiff.imwrite(out_path, thickness)
    print(f"Saved {out_path}  shape={thickness.shape}  "
          f"range=[{thickness[~missing].min() if not missing.all() else 'n/a'}, "
          f"{thickness[~missing].max() if not missing.all() else 'n/a'}]")
