import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

surgery1="D:/Xiong Lab/force-displacement/microsurgery-rm pPSM+PD/stiffness_summary.csv"
surgery2="D:/Xiong Lab/force-displacement/microsurgery-rm PD/stiffness_summary.csv"
surgery3="D:/Xiong Lab/force-displacement/microsurgery-rm NE/stiffness_summary.csv"

def make_boxplot(filename,ax):
    df=pd.read_csv(filename)
    sns.boxplot(ax=ax,x='Status', y='Stiffness', data=df,
                order=['before','after'],
                linewidth=1.5,
                 boxprops=dict(facecolor='white', edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
                 width=0.4)
    # plot individual data points
    sns.stripplot(ax=ax,x='Status', y='Stiffness', data=df, color='black', alpha=0.7, jitter=False)
    # add lines for paired data
    n=len(df.index)
    df_wide = df.pivot(index=['Date','Sample'], columns='Status', values='Stiffness')
    for i in range(n):
        ax.plot(['before', 'after'], [df_wide['before'],df_wide['after']],
                color='gray', alpha=0.6, marker='o')

    return



# Set the visual style
sns.set_theme(style='white')

# Create boxplot
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
axs=axs.flatten()
make_boxplot(surgery1,axs[0])
make_boxplot(surgery2,axs[1])
make_boxplot(surgery3,axs[2])
axs[0].set_title('Remove pPSM + PD')
axs[1].set_title('Remove PD')
axs[2].set_title('Remove NE')

# Improve layout
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
