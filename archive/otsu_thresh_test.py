import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize,medial_axis
 
img = cv.imread('E:\\Xiong Lab\\20241127-embryos\\e4-day2.tif', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(7,7),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
th4 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=51, C=-2)

combined=cv.bitwise_and(th3,th4)

# Find contour
contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Filter by x-direction length (width of bounding box)
thinnest_contour = []
ratio=0
min_width = 500  # Adjust this based on embryo body axis length
# max_width = 500  # Optional maximum limit

for i,contour in enumerate(contours):
    x, y, w, h = cv.boundingRect(contour)
    if (w > min_width) & ((w/h)>ratio):
        ratio=w/h
        thinnest_contour.append(contour)
'''
ratios.sort(reverse=True, key=lambda x: x[0])
top_3_contours = [contours[index] for _, index in ratios[:3]]
'''
all_contours=img.copy()
contour_img = img.copy()
binary_img = np.zeros_like(img)
cv.drawContours(all_contours, contours, -1, (255), thickness=cv.FILLED)
cv.drawContours(contour_img, thinnest_contour, -1, (255), thickness=cv.FILLED)
cv.drawContours(binary_img, thinnest_contour, -1, (255), thickness=cv.FILLED)

kernel = np.ones((13,13),np.uint8)
dilation= cv.dilate(binary_img,kernel,iterations = 1)
closing = cv.erode(dilation, kernel,iterations=2)

skeleton = skeletonize(closing)
# Find the indices of non-zero (white) pixels
y_coords, x_coords = np.where(skeleton > 0)

# plot all the images and their histograms
images = [img,
           closing,skeleton]
titles = ["original",
          "thinnest contour after closing","skeleton"]
 
for i in range(3):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.show()