import os
import re
import cv2 as cv
import numpy as np
import body_axis
from matplotlib import pyplot as plt

parent_folder="E:\\Xiong Lab\\embryo images\\"
folders={
    '250301':"20250301-1h bending",
    '250309':"20250309-1h bending + compression",
    '250314':"20250314-1h bending + compression",
    '250316':"20250316-1h bending + compression",
    '250406':"20250406-1h bending + compression"
    }
out_path="E:\\Xiong Lab\\axis shape data\\"

pattern = r"(?P<date>e\d+)-(?P<time>.+)\.tif"

blur_ker=7
min_width=500
opening_ker=10
fill_lo=2
fill_up=1
fit_s=5000
out_type='.csv'

bad_image=[]
for date,folder in folders.items():
    in_path=parent_folder+folder
    os.chdir(in_path)
    filelist=os.listdir()
    for filename in filelist:
        match = re.match(pattern, filename)
        if not match:
            continue
        samplename = match.group("samplename")
        time = match.group("time")
        if time=='0h':
            continue
        
        img=body_axis.load_image(filename)
        assert img is not None, "file could not be read, check with os.path.exists()"
        print('Processing',filename,'from',date)

        blur_img=body_axis.blur(img,blur_ker)
        embryo=body_axis.otsu_threshold(blur_img)
        outlines=body_axis.adaptive_threshold(blur_img)
        combined=cv.bitwise_and(embryo,outlines)

        contours,mask=body_axis.find_contours(combined)
        filtered=body_axis.filter_contour_width(contours,min_width)
        if filtered==[]:
            print('cannot segment',date,samplename)
            img_path=folder+"\\"+filename
            bad_image.append(img_path)
            continue
        notochord_contour=body_axis.find_thinnest_contour(filtered)

        notochord=np.zeros_like(img)
        cv.drawContours(notochord,[notochord_contour],-1, (255), thickness=cv.FILLED)
        notochord_opened=body_axis.opening(notochord,opening_ker)
        notochord_opened=body_axis.extend(img,notochord_opened,direction='right',lo=fill_lo,up=fill_up)
        y_ske,x_ske=body_axis.find_midaxis(notochord_opened)
        y_fit,x_fit=body_axis.fit_b_spline(y_ske,x_ske,fit_s)
        curvature=body_axis.compute_curvature(x_fit,y_fit)

        print('Finished processing',filename,'from',date,',saving data.')
        out_name=out_path+date+'-'+samplename+'-'+time
        out_table=out_name+out_type
        out_img=out_name+'.png'
        body_axis.write_coords(x_fit,y_fit,out_table,out_type)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        body_axis.plot_feature(ax,x_fit,y_fit,curvature,lim=[2560,1922],colour_lim=[-0.005,0.005])
        plt.savefig(out_img, bbox_inches='tight', dpi=300)
        plt.close()

np.savetxt(out_path+'bad_images.txt',bad_image,fmt='%s')

