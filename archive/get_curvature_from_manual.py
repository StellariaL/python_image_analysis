import cv2 as cv
import numpy as np
import pandas as pd
import body_axis
import os
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
in_path="D:\\Xiong Lab\\axis shape data\\raw manual selections\\TiFM_bending\\"
os.chdir(in_path)
samplelist=os.listdir()

samplelist=['250227-e3-before.csv','250227-e3-after1h.csv','250227-e3-day2.csv']
for sample in samplelist:
    print('Processing',sample)
    samplename=sample.removesuffix('.csv')
    date,name=samplename.split("-",1)
    manual=pd.read_csv(sample,encoding="utf-8")
    x=manual['x'].to_numpy()
    y=manual['y'].to_numpy()
    y_fit,x_fit=body_axis.fit_b_spline(y,x,100)
    curvature=body_axis.compute_curvature(x_fit,y_fit)
    print('Finished processing',sample,', saving data')

    img_path=parent_folder+folders[date]+'\\'+name+'.tif'
    out_table=out_path+samplename+'.csv'
    out_img=out_path+samplename+'.png'

    body_axis.write_coords(x_fit,y_fit,out_table)
    img=body_axis.load_image(img_path)
    fig,ax=plt.subplots(1,2)
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(img, cmap='gray')
    body_axis.plot_feature(ax[1],x_fit,y_fit,curvature,lim=[2560,1922],colour_lim=[-0.005,0.005])
    #plt.show()
    #'''
    plt.savefig(out_img, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    #'''