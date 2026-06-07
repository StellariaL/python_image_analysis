import os
import re
import numpy as np
import pandas as pd
import cv2 as cv
import scipy.odr as odr
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

markers=['o', 's', '^', 'D', '<','>','p','v','.']

def linear_model(beta, x):
    return beta[0] + beta[1] * x

def odr_fit(x,y):
    p=np.polyfit(x,y,1)
    model = odr.Model(linear_model)
    data = odr.Data(x,y)
    odrfit = odr.ODR(data, model, beta0=[p[1],p[0]])
    odr_result = odrfit.run()
    return odr_result.beta


ref_start=400
ref_length=100
end_length=50
treatment_info_path="D:\\Xiong Lab\\axis shape data\\TiFM_bending_removePSM\\info.csv"
coords_folder="D:\\Xiong Lab\\axis shape data\\TiFM_bending_removePSM\\"
pattern = re.compile(r'([0-9]+)-(e\d+)-([a-zA-Z0-9]+)\.csv')

treatment_info = pd.read_csv(treatment_info_path)
samples=treatment_info.groupby(['date','sample'])
number=samples.ngroups
summary_list=[]
no=0

#fig, axs = plt.subplots(2,3)
#axs=axs.flatten()

for filename in os.listdir(coords_folder):
    match = pattern.match(filename)
    #if not filename.startswith('250702-e4'):
        #continue
    if match:
        date=match.group(1)
        sample = match.group(2)
        timepoint = match.group(3)
        coords=pd.read_csv(os.path.join(coords_folder, filename))
        coords=coords.sort_values(by='x')
        x=coords['x'].to_numpy()
        y=coords['y'].to_numpy()
        dy = coords['y'].diff().iloc[1:]  # skip the first NaN
        dx = 1
        total_length = np.sum(np.sqrt(1 + dy**2))
        x_ref=np.array(x[ref_start:(ref_start+ref_length)])
        y_ref=np.array(y[ref_start:(ref_start+ref_length)])
        x_end=np.array(x[-end_length:])
        y_end=np.array(y[-end_length:])
        beta_ref=odr_fit(x_ref,y_ref)
        beta_end=odr_fit(x_end,y_end)
        '''
        line_ref=linear_model(beta_ref,x)
        line_end=linear_model(beta_end,x)
        
        ax=axs[no]
        ax.plot(x,y,'b')
        ax.plot(x,line_ref,'r--')
        ax.plot(x,line_end,'r--')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(filename.removesuffix('.csv'))
        no=no+1
        if no>=len(axs):
            break
        '''
        angle = np.arctan2(beta_end[1],1)-np.arctan2(beta_ref[1],1)
        row={'date':date,'sample':sample,'timepoint':timepoint,'angle':angle,'totallength':total_length}
        summary_list.append(row)
        no=no+1

summary = pd.DataFrame(summary_list)
angles=summary.pivot(index=['date','sample'],columns='timepoint',values='angle')
lengths=summary.pivot(index=['date','sample'],columns='timepoint',values='totallength')
angles['0'] = angles['0'].fillna(0)
lengths['0']=lengths['0'].fillna(0)

relative_angles=angles.sub(angles['0'],axis=0)
relative_angles=relative_angles.div(relative_angles['1'],axis=0)

elongation=lengths.sub(lengths['1'],axis=0)

x=list(relative_angles.columns)
relative_angles=relative_angles.reset_index()
relative_angles['date'] = relative_angles['date'].astype(str)
treatment_info['date'] = treatment_info['date'].astype(str)
relative_angles=relative_angles.merge(treatment_info,on=['date','sample'])
elongation=elongation.merge(treatment_info,on=['date','sample'])

t_rPSM=np.concatenate((np.array([0,1,1.1]),np.arange(2,int(x[-1]))))
t_none=np.arange(int(x[-1])+1)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
sampleno=0
# Plot each row
for _, row in relative_angles.iterrows():
    y = row[x]
    t=t_rPSM if 'rPSM' in row['treatment'] else t_none
    linestyle='-' if 'rPSM' in row['treatment'] else '--'
    color = 'red' if 'PD' in row['treatment'] else 'blue'
    plt.plot(t, y,
             color=color, alpha=0.6,
             label=row['date']+'-'+row['sample'],
             linestyle=linestyle)  # alpha for slight transparency
    sampleno=sampleno+1
    #plt.plot(x, y,label=row['date']+'-'+row['sample'])

# Aesthetics
plt.xlabel('Time (h)',fontsize=18)
plt.ylabel('Relative bending angle',fontsize=18)
plt.tick_params(axis='both', labelsize=16)
plt.axhline(y=0,color='black',linestyle='--')
#plt.legend(loc='lower left')
plt.tight_layout()


plt.subplot(1,2,2)
sampleno=0
for _, row in elongation.iterrows():
    y = row[x]
    t=t_rPSM if 'rPSM' in row['treatment'] else t_none
    linestyle='-' if 'rPSM' in row['treatment'] else '--'
    color = 'red' if 'PD' in row['treatment'] else 'blue'
    plt.plot(t, y,
             color=color, alpha=0.6,
             label=row['date']+'-'+row['sample'],
             linestyle=linestyle)  # alpha for slight transparency
    sampleno=sampleno+1

plt.xlim(left=1)
plt.ylim(bottom=-100)
plt.xlabel('Time (h)',fontsize=18)
plt.ylabel('elongation',fontsize=18)
plt.tick_params(axis='both', labelsize=16)
#plt.legend(loc='upper left')
plt.tight_layout()

plt.show()
