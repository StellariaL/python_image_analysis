import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

counts_folder='D:\\Xiong Lab\\nuclei count\\0928-MMPi\\'
roi_folder='D:\\Xiong Lab\\nuclei count\\masks\\'
probe_loc_csv='D:\\Xiong Lab\\20250928-sora\\probe_location.csv'

xres=3.6364 # pixel/um
bin_width=int(125*xres) # um converted to px

def density_map(coords,mask,bins):
    mask=mask/np.max(mask)
    counts, _ = np.histogram(coords, bins=bins)
    roi_width_per_bin = []
    for i in range(len(bins) - 1):
        y_start, y_end = bins[i], bins[i+1]
        roi_slice = mask[y_start:y_end, :]
        roi_width_per_bin.append(roi_slice.sum())  # total "area" (True pixels) in this bin
    roi_width_per_bin = np.array(roi_width_per_bin)

    density = np.zeros_like(counts, dtype=float)
    print(np.min(roi_width_per_bin))
    valid = roi_width_per_bin > 1000
    density[valid] = counts[valid] / roi_width_per_bin[valid]
    summary=pd.DataFrame(data=(bins[:-1] + bins[1:]) / 2,columns=['y'])
    summary['area']=roi_width_per_bin
    summary['counts']=counts
    summary['density']=density
    return summary



fig, axs = plt.subplots(1,3, sharex=True)
axs=axs.flatten()

probe_loc=pd.read_csv(probe_loc_csv,dtype='str')
probe_loc=probe_loc.astype({'y':'float'})
probe_loc=probe_loc.set_index(['date','sample'])

for filename in os.listdir(counts_folder):
    if filename.endswith('leftPSM-nuclei.csv'):
        name=filename.removesuffix('leftPSM-nuclei.csv')
        fields=name.split('-')
        date=fields[0]
        sample=fields[1]
        holdtime=fields[2]
        probe=probe_loc.loc[date].loc[sample,'y']
        print('processing ',[date,'-',sample])
        leftdata=pd.read_csv(counts_folder+filename)
        rightdata=pd.read_csv(counts_folder+name+'rightPSM-nuclei.csv')
        left_mask=cv.imread(roi_folder+name+'leftPSM-mask.tif',cv.IMREAD_GRAYSCALE)
        right_mask=cv.imread(roi_folder+name+'rightPSM-mask.tif',cv.IMREAD_GRAYSCALE)
        left_points=np.where(left_mask>0)
        right_points=np.where(right_mask>0)
        y_top=np.max([np.min(left_points[0]),np.min(right_points[0])])
        y_bottom=np.min([np.max(left_points[0]),np.max(right_points[0])])
        bins=np.arange(y_top,y_bottom,bin_width)
        left_coords = leftdata["centroid-1"].values
        right_coords=rightdata["centroid-1"].values
        left_summary=density_map(left_coords,left_mask,bins)
        right_summary=density_map(right_coords,right_mask,bins)
        all_summary=pd.merge(left=left_summary,right=right_summary,on='y',suffixes=('_left','_right'))
        all_summary['diff']=all_summary['density_right']-all_summary['density_left']
        all_summary['diff_relative']=(all_summary['diff']*2)/(all_summary['density_left']+all_summary['density_right'])
        
        all_summary.to_csv(counts_folder+name+'density.csv')

        all_summary['y']=all_summary['y']-probe

        marker='v' if holdtime=='0' else 'o'
        linestyle='solid' if holdtime=='60' else 'dashed'
        axs[0].plot(all_summary['y'], all_summary['density_left'], linestyle=linestyle, color='b',label=date+'-'+sample+'-left')
        axs[0].plot(all_summary['y'],all_summary['density_right'],linestyle=linestyle,color='r',label=date+'-'+sample+'-right')
        axs[1].plot(all_summary['y'],all_summary['diff'],linestyle=linestyle,label=date+'-'+sample)
        axs[2].plot(all_summary['y'],all_summary['diff_relative'],linestyle=linestyle,label=date+'-'+sample)

axs[0].set_ylabel('cell density')
axs[1].axhline(color='k')
axs[1].legend()
axs[2].axhline(color='k')
plt.show()
