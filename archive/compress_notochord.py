#under construction

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
imgpath="C:\\Users\\ruohe\\Xiong Lab\\compression_test\\a(1).tif"
roipath="C:\\Users\\ruohe\\Xiong Lab\\compression_test\\roi.csv"
probex=239
probewidth=50
top=0
left=0
right=500

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# --- Step 1: Load image in grayscale and color ---
gray_img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
gray_img = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE)
gray_img=gray_img[top:,left:right]
m1=np.max(gray_img)
m2=np.min(gray_img)
enhanced=((gray_img-m2)/(m1-m2))*255
enhanced=enhanced.astype('uint8')
color_img = cv2.imread(imgpath)  # for contour overlay
color_img=cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
roi_points = pd.read_csv(roipath)  # expects columns 'x' and 'y'
pts = roi_points[['X', 'Y']].to_numpy().astype(np.int32)

probe=gray_img[:,probex:(probex+probewidth)]
probe=probe.astype(float)
line=np.sum(probe,axis=1)
line=moving_average(line,4)
linediff=np.diff(line)
probepos=np.argmin(linediff[10:])+10+top

# --- Step 4: Apply adaptive thresholding ---
ret3,thresh = cv2.threshold(enhanced,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest=None
area=0
for contour in contours:
    if cv2.contourArea(contour)>area:
        area=cv2.contourArea(contour)
        largest=contour
mask = np.zeros_like(enhanced)
cv2.drawContours(mask, largest, -1, (255), thickness=cv2.FILLED)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

opened=mask

# --- Step 5: Find contours ---
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Step 6: Draw contours on color image ---
contour_img = color_img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# --- Step 7: Calculate and plot contour area histogram ---
areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0]

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title('Contours Overlay')

plt.subplot(1, 3, 2)
plt.hist(areas, bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Contour Areas')
plt.xlabel('Area')
plt.ylabel('Count')

plt.subplot(1,3,3)
plt.imshow(opened)

plt.tight_layout()
plt.show()
