import cv2 as cv
import numpy as np
import body_axis
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt

parent_folder="E:\\Xiong Lab\\embryo images\\"
folders={
    '241115':"20241115-Fgf inhibition 4uM & TiFM",
    '241121':"20241121-Fgf inhibition 4uM + cytoGFP & TiFM",
    '241124':"20241124-Fgf inhibition 4uM & TiFM",
    '241127':"20241127-Fgf inhibition 4uM & TiFM",
    '241202':"20241202-Fgf inhibition 4uM +cytoGFP & TiFM",
    '241208':"20241208-Fgf inhibition 4uM & TiFM"
    }
out_path="E:\\Xiong Lab\\axis shape data\\"

date='241115'
sample='e10'

img_path=parent_folder+folders[date]+'\\'+sample+'-day2.tif'
out_table=out_path+date+'-'+sample+'-day2.csv'
out_img=out_path+date+'-'+sample+'-day2.png'

img=body_axis.load_image(img_path)
assert img is not None, "file could not be read, check with os.path.exists()"

blur_img=body_axis.blur(img,7)
embryo=body_axis.otsu_threshold(blur_img)
outlines=body_axis.adaptive_threshold(blur_img)
combined=cv.bitwise_and(embryo,outlines)

contours,mask=body_axis.find_contours(combined)

filtered=body_axis.filter_contour_width(contours,500)
filtered_contours=np.zeros_like(img)
cv.drawContours(filtered_contours,filtered,-1, (255), thickness=cv.FILLED)
notochord_contour=body_axis.find_thinnest_contour(filtered)

#'''
notochord=np.zeros_like(img)
cv.drawContours(notochord,[notochord_contour],-1, (255), thickness=cv.FILLED)
#body_axis.show_images([img,notochord])
#'''
notochord_opened=body_axis.opening(notochord,10)
notochord_opened=body_axis.extend(img,notochord_opened,direction='right',lo=2,up=1,ker=20)
#notochord_opened=body_axis.opening(notochord,14)

y_ske,x_ske=body_axis.find_midaxis(notochord_opened)
skeleton=skeletonize(notochord_opened)
y_fit,x_fit=body_axis.fit_b_spline(y_ske,x_ske,9000)

curvature=body_axis.compute_curvature(x_fit,y_fit)

body_axis.show_images([embryo,outlines,combined],["otsu thresholding","adaptive thresholding","combined"])
body_axis.show_images([filtered_contours,notochord,notochord_opened,skeleton],["filtered contours","notochord","final","skeleton"])
#'''
fig,ax=plt.subplots(1,2)
ax[0].imshow(img, cmap='gray')
ax[1].imshow(img, cmap='gray')
body_axis.plot_feature(ax[1],x_fit,y_fit,curvature,lim=[2560,1922],colour_lim=[-0.005,0.005])
plt.show()

