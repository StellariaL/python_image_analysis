import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

filename='E:\\PhD_large_images\\20260410\\buckling_raw2.csv'
ctrl_tag='ctrl'
exp_tag='rocki'

df=pd.read_csv(filename)
df['displacement']=df['rightprobe']-df['fold1']
df['distance']=df['buckle']-df['leftprobe']

print('statistical test for displacement:')
ctrl=df.loc[df['treatment']==ctrl_tag,['displacement']]
exp=df.loc[df['treatment']==exp_tag,['displacement']]

t_stat, t_pval = stats.ttest_ind(a=ctrl, b=exp, alternative="two-sided")
print(f"t-test: t = {t_stat}, p = {t_pval}")
u_stat, u_pval = stats.mannwhitneyu(ctrl,exp, alternative='two-sided')
print(f"MW test: U = {u_stat}, p = {u_pval}")

print('statistical test for distance:')
ctrl=df.loc[df['treatment']==ctrl_tag,['distance']]
exp=df.loc[df['treatment']==exp_tag,['distance']]

t_stat, t_pval = stats.ttest_ind(a=ctrl, b=exp, alternative="two-sided")
print(f"t-test: t = {t_stat}, p = {t_pval}")
u_stat, u_pval = stats.mannwhitneyu(ctrl,exp, alternative='two-sided')
print(f"MW test: U = {u_stat}, p = {u_pval}")

'''
print(df.head())
'''
# Set the visual style
sns.set_theme(style='white')

# Create boxplot
fig, axs = plt.subplots(2, 1, sharex=True)
axs=axs.flatten()
sns.boxplot(x='treatment', y='displacement', data=df,ax=axs[0],
                 linewidth=1.5,
                 boxprops=dict(edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
                 width=0.4)
# plot individual data points
sns.stripplot(x='treatment', y='displacement',data=df, ax=axs[0], color='black',alpha=0.7, jitter=False)
axs[0].set_xlabel('')
axs[0].set_ylabel('buckling displacement (um)',fontsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=14)

sns.boxplot(x='treatment', y='distance', data=df,ax=axs[1],
            linewidth=1.5,
            boxprops=dict(edgecolor='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='black'),
            flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
            width=0.4)
sns.stripplot(x='treatment', y='distance',data=df, ax=axs[1], color='black',alpha=0.7, jitter=False)
axs[1].set_xlabel('')
axs[1].set_ylabel('buckle distance to left probe (um)',fontsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=14)
             
# Improve layout
plt.tight_layout()
plt.show()
#'''