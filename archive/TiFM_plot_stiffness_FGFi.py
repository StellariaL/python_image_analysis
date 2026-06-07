import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

filename='D:/Xiong Lab/force-displacement/FGFi-overnight-summary.csv'
ctrl_tag='DMSO'
exp_tag='PD'

raw=pd.read_csv(filename)
df=raw.loc[raw['Quality']=='good',:]

ctrl=df.loc[df['Treatment']==ctrl_tag,['Stiffness']]
exp=df.loc[df['Treatment']==exp_tag,['Stiffness']]

t_stat, t_pval = stats.ttest_ind(a=ctrl, b=exp, alternative="two-sided")
print(f"t-test: t = {t_stat}, p = {t_pval}")
u_stat, u_pval = stats.mannwhitneyu(ctrl,exp, alternative='two-sided')
print(f"MW test: U = {u_stat}, p = {u_pval}")


# Set the visual style
sns.set_theme(style='white')

# Create boxplot
plt.figure(figsize=(6, 4))
ax = sns.boxplot(x='Treatment', y='Stiffness', data=df,
                 order=[ctrl_tag, exp_tag],
                 linewidth=1.5,
                 boxprops=dict(facecolor='white', edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
                 width=0.4)

# plot individual data points
sns.stripplot(x='Treatment', y='Stiffness', data=df, color='black', alpha=0.7, jitter=True)


plt.tight_layout()
plt.show()
