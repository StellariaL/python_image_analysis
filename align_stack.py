import numpy as np
import tifffile
import os
from roifile import ImagejRoi, roiread
from scipy.ndimage import center_of_mass, shift
import imageio

# === Input ===
tiff_path = "E:\\PhD_large_images\\20260423\\stacks\\3-2_stack.tif"
ref_stack_path="E:\\PhD_large_images\\20260423\\3-2-alignment_mask.tif"
#roi_zip_path = "D:\\Xiong Lab\\20250518\\notochord_alignment_e1_live.zip"
output_path = "E:\\PhD_large_images\\20260423\\stacks\\3-2-aligned.tif"

# === Load the image stack ===
stack = tifffile.imread(tiff_path)
ref=tifffile.imread(ref_stack_path)
num_slices = stack.shape[0]
'''
# === Load ROIs ===
rois = roiread(roi_zip_path)

# === Get x-centers of the rectangle ROIs ===
x_centers = []
for roi in rois:
    left = roi.left
    right = roi.right
    x_center = (left+right) / 2
    x_centers.append(x_center)

x_centers = np.array(x_centers)

# === Translate each slice accordingly ===
aligned_stack = []
for i in range(num_slices):
    slice_img = stack[i]
    shift_x = target_center - x_centers[i]
    shifted_img = shift(slice_img, shift=(0, shift_x), order=1, mode='nearest')
    aligned_stack.append(shifted_img)
'''
centroids = np.array([center_of_mass(frame) for frame in ref])
# === Define target alignment position ===
#target_center = np.mean(x_centers)  # Or choose a specific value
target_center = centroids[0]
raw_shifts = target_center - centroids      # (dy, dx)
shifts = np.zeros_like(raw_shifts)
shifts[:, 1] = raw_shifts[:, 1]   # keep only dx
aligned_stack = np.array([
    shift(stack[i], shift=shifts[i], order=1)
    for i in range(len(stack))
])

aligned_stack = np.array(aligned_stack, dtype=stack.dtype)

# === Save the result ===
tifffile.imwrite(output_path, aligned_stack)
print(f"Aligned stack saved to: {output_path}")