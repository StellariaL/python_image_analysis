import os
import numpy as np
import cv2 as cv
import pandas as pd
from skimage import measure

seg_folder='D:\\Xiong Lab\\nuclei count\\0928-ctrl\\'
mask_folder='D:\\Xiong Lab\\nuclei count\\masks\\'
output_folder='D:\\Xiong Lab\\nuclei count\\'


for filename in os.listdir(seg_folder):
    #if filename.endswith('-nuclei_seg.npy'):
    if filename.endswith('250928-e10-60-leftPSM-nuclei_seg.npy'):
        sample=filename.removesuffix("-nuclei_seg.npy")
        print('processing '+sample)
        roifile=sample+"-mask.tif"
        roi_mask=cv.imread(mask_folder+roifile, cv.IMREAD_GRAYSCALE)
        y_coords, x_coords = np.where(roi_mask > 0)
        min_y=np.min(y_coords)
        min_x=np.min(x_coords)

        segmentation= np.load(seg_folder+filename, allow_pickle=True).item()
        masks=segmentation["masks"]
        props = measure.regionprops_table(masks, properties=['centroid'])
        props_df = pd.DataFrame(props)
        props_df=props_df.drop(columns=['centroid-0'])
        props_df['centroid-1']=props_df['centroid-1']+min_y
        props_df['centroid-2']=props_df['centroid-2']+min_x

        outname=sample+'-nuclei.csv'
        props_df.to_csv(output_folder+outname)
        print('finished processing '+sample)
        