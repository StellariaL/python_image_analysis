'''
This script takes a 3D stack and generates an 2D projection of its surface layer. Good when the surface is curved.

workflow:
1. read stack
2. apply mean filter to each slice
3. for each pixel:
    find the first z slice in the blurred stack that exceeds an intensity threshold;
    then, take the next few z slices and take a max projection

input parameters:
    input_path: string, path to the stack; needs to be .tif
    r_blur: int, mean filter radius; take a value large enough to cover your characteristic feature
    threshold: int, intensity value threshold to find surface
    zrange: int, number of slices taken for max projection; try a small value

outputs:
    *-surf.tif: 2D surface projection; same XY size as input stack
    *-record.tif: 3D mask (16bit), pixel taken by the projection is coloured white; same XYZ size as input stack.

tips:
Open the -record mask in fiji and merge it to the input stack to visualize which pixels are taken.

'''


import numpy as np
import tifffile as tiff
import cv2
import re
import os
from scipy.ndimage import median_filter

folder="E:\\PhD_large_images\\20260512-cellshape\\"
r_blur=200
half_window=3
peak_percentage=0.8
smooth_ker=200

zrange=2*half_window+1
g_blur_k=2*r_blur+1

pattern = re.compile(r'0603-e[0-9]-actin-[0-9]\.tif')
#pattern=re.compile(r'0603-e2-actin-1\.tif')

for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        input_path=folder+filename
        # outputs
        projection_path=input_path.removesuffix('.tif')+'-surf.tif'
        record_path=input_path.removesuffix('.tif')+'-record.tif'

        # initializations
        stack = tiff.imread(input_path)
        if stack.ndim != 3:
            raise ValueError("Input image must be a grayscale z-stack (Z, Y, X).")

        Z, Y, X = stack.shape

        mask=np.zeros((Z,Y, X), dtype=stack.dtype)
        blurred=np.zeros((Z,Y, X), dtype=stack.dtype)
        local_z=np.zeros((Y,X),dtype=stack.dtype)
        projection = np.zeros((Y, X), dtype=stack.dtype)
        final_z=np.zeros((Z,Y,X),dtype=stack.dtype)
        window = (np.arange(zrange) - half_window).reshape(zrange, 1, 1)

        # blur image and threshold
        for z in range(Z):
            # cv2.GaussianBlur(src, ksize, sigmaX, sigmaY)
            # Setting sigmaY=0 defaults it to match sigmaX (r_blur)
            blurred[z, :, :] = cv2.GaussianBlur(
                stack[z, :, :], 
                (g_blur_k, g_blur_k), 
                sigmaX=r_blur
            ).astype(np.float32)
        
        # surface is assumed to be at steepest z intensity gradient
        threshold_z = np.zeros((Y, X), dtype=np.int32)
        
        for y in range(Y):
            for x in range(X):
                z_profile = blurred[:, y, x]
                peak_z = np.argmax(z_profile)
                peak_intensity = z_profile[peak_z]
                
                local_threshold = peak_intensity *peak_percentage
                
                # Find the first Z slice that exceeds local threshold
                # We restrict the search from 0 up to the peak itself
                above_threshold_indices = np.where(z_profile[:peak_z + 1] >= local_threshold)[0]
                
                if len(above_threshold_indices) > 0:
                    threshold_z[y, x] = above_threshold_indices[0]
                else:
                    threshold_z[y, x] = peak_z

        smooth_z = median_filter(threshold_z, size=smooth_ker)
        surf_z_idx = np.round(smooth_z).astype(np.int32)
        
        # create centered window indices around the steepest gradient slice
        local_z = surf_z_idx[np.newaxis, :, :] + window
        local_z = np.clip(local_z, 0, Z - 1)

        y_ind = np.broadcast_to(np.arange(Y)[:, None], (zrange, Y, X))
        x_ind = np.broadcast_to(np.arange(X)[None, :], (zrange, Y, X))
        
        local_intensities = stack[local_z, y_ind, x_ind]
        
        # Find local maximum within the window and map back to absolute Z indices
        local_max_offset = np.argmax(local_intensities, axis=0)
        
        # Reconstruct the true absolute Z coordinate for the max intensity pixel
        # local_z shape is (zrange, Y, X); we index it using the offset
        y_mesh, x_mesh = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
        absolute_max_z = local_z[local_max_offset, y_mesh, x_mesh]
        
        # Build the final 2D projection
        projection = stack[absolute_max_z, y_mesh, x_mesh]
        
        # Record the chosen pixel in the 3D mask
        # Value 4096 assumes 12-bit/16-bit range data; adjust if using 8-bit (255)
        final_z[absolute_max_z, y_mesh, x_mesh] = 4096

        # print and save output
        tiff.imwrite(projection_path, projection)
        tiff.imwrite(record_path,final_z)