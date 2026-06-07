'''
Template for making box plots
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

filename=''

df=pd.read_csv(filename)


# Set the visual style
sns.set_theme(style='white',font_scale=2) # default font scale is 1

# Create boxplot
plt.figure(figsize=(6, 4))
ax = sns.boxplot(x='group', y='value', data=df,
                 order=['1', '2', '3'],
                 linewidth=1.5,
                 boxprops=dict(facecolor='white', edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
                 width=0.4)

# plot individual data points
sns.stripplot(x='group', y='value', data=df, color='black', alpha=0.7, jitter=False)

# add lines for paired data
n=len(df.index)
df_wide = df.pivot(index='sample', columns='group', values='value')
for i in range(n):
    plt.plot(['group1', 'group2', 'group3'], [df_wide['group1'],df_wide['group2'],df_wide['group3']],
             color='gray', alpha=0.6, marker='o')

# Improve layout
plt.ylim(-0.1,0.5)
plt.title('')
plt.tight_layout()
plt.show()
