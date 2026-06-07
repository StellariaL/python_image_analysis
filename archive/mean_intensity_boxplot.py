import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

raw=pd.read_excel("D:/Xiong Lab/20250628-confocal/projections/SUM_roi_intensity_results.xlsx")

NNE_data=raw[raw['tissue'] =="NNE"]
NNE_data=NNE_data.pivot(index=['treatment','sample','roi'],columns=['channel'],values='mean_intensity')
NNE_data['normalized']=NNE_data[3]/NNE_data[1]

NP_data=raw[raw['tissue'] =="NP"]
NP_data=NP_data.pivot(index=['treatment','sample','roi'],columns=['channel'],values='mean_intensity')
NP_data['normalized']=NP_data[3]/NP_data[1]

fig,axes=plt.subplots(1,2)

sns.boxplot(x='treatment', y='normalized', hue='sample',data=NNE_data,ax=axes[0],
                 linewidth=1.5,width=0.4)
sns.boxplot(x='treatment', y='normalized', hue='sample',data=NP_data,ax=axes[1],
                 linewidth=1.5,width=0.4)

axes[0].set_title('non-neural ectoderm')
axes[0].legend(loc='best')
axes[1].set_title('neural plate')
axes[1].legend(loc='best')

plt.tight_layout()
plt.show()
