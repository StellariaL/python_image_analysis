import os
import numpy as np
import pandas as pd
import cv2 as cv
import scipy.odr as odr
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

timepoints={'before':0,
            '0h':1,
            '1h':2,
            '2h':3,
            '3h':4,
            '4h':5,
            '5h':6,
            '6h':7}


ref_length=500 # the point this many pixel away from the endpoint will be aligned to the x axis.
ref_ind=ref_length

treatment_info_path="D:\\Xiong Lab\\20250702\\info.csv"
raw_folder="D:\\Xiong Lab\\20250702\\notochord_skeleton_image\\"
output_folder="D:\\Xiong Lab\\20250702\\notochord_skeleton_coords\\"

treatment_info = pd.read_csv(treatment_info_path)

samples=treatment_info.groupby(['date','sample'])
number=samples.ngroups
colors = plt.cm.turbo(np.linspace(0, 1, treatment_info["time"].nunique()))
rows = int(np.ceil(np.sqrt(number)))
cols = int(np.ceil(number / rows))
fig, axs = plt.subplots(rows, cols)
axs=axs.flatten()

no=0

for name,group in samples:
    samplename=str(name[0])+'-'+name[1]
    ax=axs[no]
    for i,row in group.iterrows():
        filename = f"{samplename}-{row['time']}.tif"
        filepath = os.path.join(raw_folder, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        try:
            scalebar = ScaleBar(1.8, "um", length_fraction=0.2,
                                location='lower left',
                                frameon=False,scale_loc='none')
            ax.add_artist(scalebar)
            skeleton=cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            y_coords, x_coords = np.where(skeleton > 0)
            dataframe=pd.DataFrame({'x':x_coords,'y':y_coords})
            dataframe=dataframe.sort_values(by='x')
            name=output_folder+filename.removesuffix('.tif')+'.csv'
            dataframe.to_csv(name,index=False,sep=',')
            x_aligned = x_coords - min(x_coords)
            '''
            x_fitting=np.array(x_aligned[:ref_ind])
            y_fitting=np.array(y[:ref_ind])
            p=np.polyfit(x_fitting,y_fitting,1)
            model = odr.Model(linear_model)
            data = odr.Data(x_fitting,y_fitting)
            odrfit = odr.ODR(data, model, beta0=[p[1],p[0]])
            odr_result = odrfit.run()
            beta=odr_result.beta
            y_aligned = y - beta[0]
            angle = -np.arctan2(-beta[1], -1)
            x_rotated = x_aligned * np.cos(angle) - y_aligned * np.sin(angle)
            y_rotated = x_aligned * np.sin(angle) + y_aligned * np.cos(angle)-50*timepoints[row['time']]
            '''
            x_plot=x_aligned
            y_plot=y_coords-min(y_coords)+150*timepoints[row['time']]
            ax.scatter(x_plot,-y_plot,marker=".",color=colors[timepoints[row['time']]])
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    ax.set_title(samplename+'-'+row['treatment'])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    no+=1

plt.suptitle('Aligned Notochord Shapes')
plt.tight_layout()
plt.show()
