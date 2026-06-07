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
import matplotlib.pyplot as plt

folder="E:\\PhD_large_images\\20260512-cellshape\\"
r_blur=50
threshold=200
zrange=5

#pattern = re.compile(r'(e5)-40x-view(\d+)-actin\.tif')
pattern = re.compile(r'.*-view4\.tif')

for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        input_path=folder+filename
        # outputs
        projection_path=input_path.removesuffix('.tif')+'-surf.tif'
        projection_max_path=input_path.removesuffix('.tif')+'-surf_max.tif'
        record_path=input_path.removesuffix('.tif')+'-record.tif'
        record_max_path=input_path.removesuffix('.tif')+'-record_max.tif'

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
        final_z_max=np.zeros((Z,Y,X),dtype=stack.dtype)
        window=np.arange(zrange).reshape(zrange,1,1)

        # blur image and threshold
        for z in range(Z):
            temp=cv2.blur(stack[z,:,:],(r_blur,r_blur))
            blurred[z,:,:]=temp
            mask[z,:,:]=temp>threshold
            

        # find surface
        first_z = np.argmax(mask, axis=0)
        max_z=np.argmax(blurred,axis=0)

        # local max projection
        local_z=first_z[np.newaxis,:,:]+window
        local_z=np.clip(local_z,0,Z-1)
        y_ind=np.broadcast_to(np.arange(Y)[:,None],(zrange,Y,X))
        x_ind=np.broadcast_to(np.arange(X)[None,:],(zrange,Y,X))
        local_intensities=stack[local_z,y_ind,x_ind]
        local_max=np.argmax(local_intensities,axis=0)+first_z
        y_ind=np.broadcast_to(np.arange(Y)[:,None],(Y,X))
        x_ind=np.broadcast_to(np.arange(X)[None,:],(Y,X))
        projection=stack[local_max,y_ind,x_ind]
        projection_max=stack[max_z,y_ind,x_ind]
        final_z[local_max,y_ind,x_ind]=4096
        final_z_max[max_z,y_ind,x_ind]=4096

        # print and save output
        plt.imshow(projection_max, cmap='gray')
        plt.show()
        tiff.imwrite(projection_path, projection)
        tiff.imwrite(projection_max_path,projection_max)
        tiff.imwrite(record_path,final_z)
        tiff.imwrite(record_max_path,final_z_max)