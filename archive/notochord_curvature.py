import cv2 as cv
import numpy as np
import body_axis
from matplotlib import pyplot as plt

parent_folder="D:\\Xiong Lab\\embryo images\\"
folders={
    '250224':"20250224-1h bending",
    '250227':"20250227-1h bending",
    '250301':"20250301-1h bending",
    '250309':"20250309-1h bending + compression",
    '250314':"20250314-1h bending + compression",
    '250316':"20250316-1h bending + compression",
    '250406':"20250406-1h bending + compression"
    }
out_path="D:\\Xiong Lab\\axis shape data\\"

date='250227'
sample='e3'
suffix='day2'

img_path=parent_folder+folders[date]+'\\'+sample+'-'+suffix+'.tif'
out_table=out_path+date+'-'+sample+'-'+suffix+'.csv'
out_img=out_path+date+'-'+sample+'-'+suffix+'.png'

img=body_axis.load_image(img_path)
assert img is not None, "file could not be read, check with os.path.exists()"

blur_img=body_axis.blur(img,7)
embryo=body_axis.otsu_threshold(blur_img)
outlines=body_axis.adaptive_threshold(blur_img,block_size=67)
combined=cv.bitwise_and(embryo,outlines)

contours,mask=body_axis.find_contours(combined)
'''
widths=[]
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)  # Get bounding rectangle
    widths.append(w)
plt.hist(widths, bins=20, color='blue', edgecolor='black')
plt.show()
'''
notochord=np.zeros_like(img)

filtered_contours=body_axis.filter_contour_width(contours,500)
cv.drawContours(notochord,filtered_contours,-1, (255), thickness=cv.FILLED)
body_axis.show_images([img,notochord])

filtered=body_axis.filter_contour_width(contours,500)
notochord_contour=body_axis.find_thinnest_contour(filtered)


notochord=np.zeros_like(img)
cv.drawContours(notochord,[notochord_contour],-1, (255), thickness=cv.FILLED)
#body_axis.show_images([img,notochord])

notochord_opened=body_axis.opening(notochord,8)
'''
notochord_opened=body_axis.extend(img,notochord_opened,direction='left',lo=10,up=1.8,ker=15)
notochord_opened=body_axis.extend(img,notochord_opened,direction='right',lo=2,up=1,ker=20)

#notochord_opened=body_axis.opening(notochord,14)
notochord_opened[:719, :] = 0
notochord_opened[919:, 1463:1497] = 0
'''
notochord_opened[:, 1950:] = 0
y_ske,x_ske=body_axis.find_midaxis(notochord_opened)
y_fit,x_fit=body_axis.fit_b_spline(y_ske,x_ske,9000)
'''
x_end=2005
x_fit=x_fit[0:(x_end-x_fit[0])]
y_fit=y_fit[0:(x_end-x_fit[0])]
'''
curvature=body_axis.compute_curvature(x_fit,y_fit)
#body_axis.show_images([img,notochord_opened],["original","notochord"])

fig,ax=plt.subplots(1,2)
ax[0].imshow(img, cmap='gray')
ax[1].imshow(img, cmap='gray')
body_axis.plot_feature(ax[1],x_fit,y_fit,curvature,lim=[2560,1922],colour_lim=[-0.005,0.005])
plt.show()

plt.savefig(out_img, bbox_inches='tight', dpi=300)
plt.close()
body_axis.write_coords(x_fit,y_fit,out_table)
