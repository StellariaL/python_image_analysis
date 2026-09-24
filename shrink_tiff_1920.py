'''
Shrink 1920×1440 px TIFF images to 960×720 px.

Iterates all .tif/.tiff files in a folder. Images whose width and height are
1920 and 1440 (2D or Z-stack with those spatial dimensions) are downscaled
by 2× and written back to the same path.
'''

import os

import cv2
import numpy as np
import tifffile as tiff

folder = "D:\\Ruoheng_Li\\20260703-ROCKibead\\"

SRC_WIDTH = 1920
SRC_HEIGHT = 1440
DST_WIDTH = 960
DST_HEIGHT = 720


def spatial_shape(img: np.ndarray) -> tuple[int, int]:
    if img.ndim == 2:
        return img.shape[1], img.shape[0]
    if img.ndim == 3 and img.shape[2] in (3, 4):
        return img.shape[1], img.shape[0]
    return img.shape[-1], img.shape[-2]


def resize_image(img: np.ndarray) -> np.ndarray:
    size = (DST_WIDTH, DST_HEIGHT)
    if img.ndim == 2:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if img.ndim == 3 and img.shape[2] in (3, 4):
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return np.stack(
        [cv2.resize(slice_img, size, interpolation=cv2.INTER_AREA) for slice_img in img],
        axis=0,
    )


for filename in os.listdir(folder):
    if not filename.lower().endswith(('.tif', '.tiff')):
        continue

    path = os.path.join(folder, filename)
    img = tiff.imread(path)
    width, height = spatial_shape(img)

    if width != SRC_WIDTH or height != SRC_HEIGHT:
        print(f"skip {filename}: {width}×{height}")
        continue

    resized = resize_image(img)
    tiff.imwrite(path, resized)
    print(f"resized {filename}: {SRC_WIDTH}×{SRC_HEIGHT} -> {DST_WIDTH}×{DST_HEIGHT}")
