import cv2 as cv
import numpy as np
import pandas as pd
import os
from skimage.morphology import skeletonize

input_path = "D:/Xiong Lab/20251223/skeletons/"  # <- Change this to your folder path
output_path="D:\\Xiong Lab\\axis shape data\\TiFM_bending_removePSM\\251223-"


# Loop through all files in the folder
for filename in os.listdir(input_path):
    if filename.endswith('.tif'):
        print("processing "+filename)
        filepath=input_path+filename
        skeleton=cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        #skeleton = skeletonize(image)
        # Find the indices of non-zero (white) pixels
        y_coords, x_coords = np.where(skeleton > 0)
        dataframe=pd.DataFrame({'x':x_coords,'y':y_coords})
        name=output_path+filename.removesuffix('.tif')+'.csv'
        dataframe.to_csv(name,index=False,sep=',')
        print("finished processing "+filename)