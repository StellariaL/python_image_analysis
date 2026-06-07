import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

projection_folder='D:\\Xiong Lab\\20250928-sora\\'
roi_folder='D:\\Xiong Lab\\nuclei count\\masks\\'
probe_loc_csv='D:\\Xiong Lab\\20250928-sora\\probe_location.csv'

xres=3.6364 # pixel/um
bin_width=int(100*xres) # um converted to px

def intensity_map(img,mask,bins,top):
    mask=mask/np.max(mask)
    roi_width_per_bin = []
    sum_intensity=[]
    for i in range(len(bins) - 1):
        y_start, y_end = bins[i], bins[i+1]
        roi_slice = mask[y_start:y_end, :]
        img_slice=img[(y_start-top):(y_end-top),:]
        roi_width_per_bin.append(roi_slice.sum())  # total "area" (True pixels) in this bin
        sum_intensity.append(img_slice.sum())
    roi_width_per_bin = np.array(roi_width_per_bin)
    sum_intensity=np.array(sum_intensity)

    intensity=np.full_like(sum_intensity,np.nan, dtype=float)
    valid = roi_width_per_bin > 1000
    intensity[valid] = sum_intensity[valid] / roi_width_per_bin[valid]
    summary=pd.DataFrame(data=(bins[:-1] + bins[1:]) / 2,columns=['y'])
    summary['area']=roi_width_per_bin.astype(float)
    summary['sum']=sum_intensity.astype(float)
    summary['intensity']=intensity.astype(float)
    return summary




fig, axs = plt.subplots(1,3, sharex=True)
axs=axs.flatten()

probe_loc=pd.read_csv(probe_loc_csv,dtype='str')
probe_loc=probe_loc.astype({'y':'float'})
probe_loc=probe_loc.set_index(['date','sample'])

for filename in os.listdir(projection_folder):
    if filename.endswith('leftPSM-fib-maxproj.tif'):
        name=filename.removesuffix('leftPSM-fib-maxproj.tif')
        fields=name.split('-')
        date=fields[0]
        sample=fields[1]
        holdtime=fields[2]
        probe=probe_loc.loc[date].loc[sample,'y']
        treatment=probe_loc.loc[date].loc[sample,'treatment']
        print('processing ',[date,'-',sample])
        left_proj=cv.imread(projection_folder+filename,cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
        right_proj=cv.imread(projection_folder+name+'rightPSM-fib-maxproj.tif',cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
        a=np.max(left_proj)
        left_mask=cv.imread(roi_folder+name+'leftPSM-mask.tif',cv.IMREAD_GRAYSCALE)
        right_mask=cv.imread(roi_folder+name+'rightPSM-mask.tif',cv.IMREAD_GRAYSCALE)
        left_points=np.where(left_mask>0)
        right_points=np.where(right_mask>0)
        left_top=np.min(left_points[0])
        right_top=np.min(right_points[0])
        y_top=np.max([left_top,right_top])
        y_bottom=np.min([np.max(left_points[0]),np.max(right_points[0])])
        bins=np.arange(y_top,y_bottom,bin_width)

        left_summary=intensity_map(left_proj,left_mask,bins,left_top)
        right_summary=intensity_map(right_proj,right_mask,bins,right_top)
        all_summary=pd.merge(left=left_summary,right=right_summary,on='y',suffixes=('_left','_right'))
        all_summary['diff']=all_summary['intensity_right']-all_summary['intensity_left']
        all_summary['diff_relative']=(all_summary['diff']*2)/(all_summary['intensity_left']+all_summary['intensity_right'])
        
        all_summary.to_csv(projection_folder+name+'fib_intensity.csv')

        all_summary['y']=all_summary['y']-probe

        marker='v' if treatment=='MMPi' else 'o'
        linestyle='solid' if holdtime=='60' else 'dashed'
        axs[0].plot(all_summary['y'], all_summary['intensity_left'], marker=marker,linestyle=linestyle, color='b',label=date+'-'+sample+'-left')
        axs[0].plot(all_summary['y'],all_summary['intensity_right'],marker=marker,linestyle=linestyle,color='r',label=date+'-'+sample+'-right')
        axs[1].plot(all_summary['y'],all_summary['diff'],marker=marker,linestyle=linestyle,label=date+'-'+sample)
        axs[2].plot(all_summary['y'],all_summary['diff_relative'],marker=marker,linestyle=linestyle,label=date+'-'+sample)

axs[0].set_ylabel('Fibronectin intensity')
axs[1].axhline(color='k')
axs[1].legend()
axs[2].axhline(color='k')
plt.show()
