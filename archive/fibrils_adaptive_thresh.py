import cv2 as cv
import numpy as np
import os
import re
from matplotlib import pyplot as plt

input_folder="D:\\Xiong Lab\\filaments\\maxprojections\\"
pattern = re.compile(r'([0-9]+)-(e\d+)-([0-9]+)-([0-9]+)-fib-maxproj\.tif')
# filename format: date-sample-holdtime-no-channelname-maxproj.tif
# eg. 250913-e3-0-1-fib-maxproj.tif
flag=0
for filename in os.listdir(input_folder):
    if flag:
        break
    flag=1
    match = pattern.match(filename)
    if match:
        date=match.group(1)
        sample = match.group(2)
        holdtime = match.group(3)
        no=match.group(4)
        img=cv.imread(input_folder+filename, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        th_mean=cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,-0.4)
        th_gauss=cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,0)
        titles = ['Original', 'Adaptive Mean Thresholding',
                  'Adaptive Gaussian Thresholding']
        images=[img,th_mean,th_gauss]
        for i in range(3):
            plt.subplot(1,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()