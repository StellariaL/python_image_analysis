import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_path="D:\\Xiong Lab\\nuclei count\\rois-0928-MMPi\\summary.csv"

raw=pd.read_csv(data_path)

data=raw.pivot(index=['date','sample','holdtime'],columns='location',values='counts')
data['diff']=data['right']-data['left']
data['diff_relative']=(data['diff']*2)/(data['right']+data['left'])


# Set the visual style
sns.set_theme(style='white')

# Create boxplot
fig, axs = plt.subplots(1, 2)
axs=axs.flatten()
sns.boxplot(ax=axs[0],x='holdtime', y='diff', data=data,
                 order=['0', '60'],
                 linewidth=1.5,
                 boxprops=dict(facecolor='white', edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
                 width=0.4)
sns.stripplot(ax=axs[0],x='holdtime', y='diff', data=data, color='black', alpha=0.7, jitter=False)
sns.boxplot(ax=axs[1],x='holdtime', y='diff_relative', data=data,
                 order=['0', '60'],
                 linewidth=1.5,
                 boxprops=dict(facecolor='white', edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
                 width=0.4)
sns.stripplot(ax=axs[1],x='holdtime', y='diff_relative', data=data, color='black', alpha=0.7, jitter=False)
# plot individual data points

# Improve layout
plt.title('right-left difference at probe level')
plt.tight_layout()
plt.show()
