import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

filename='E:\\PhD_large_images\\20260410\\buckling_raw.csv'

df=pd.read_csv(filename)
df=df.drop(columns=["notes"])
df=df.dropna()

# calculate displacements
cols = ["fold1", "unfold1", "fold2","unfold2"]
df[cols] = -df[cols].sub(df["start"], axis=0)
df=df.drop(columns=["start"])

df_wide=df
# transform to long format
df=df.melt(id_vars=["date","no","treatment"],value_vars=cols)
'''
print(len(df_wide))
'''
# Set the visual style
sns.set_theme(style='white')

# Create boxplot
plt.figure(figsize=(6, 4))
ax = sns.boxplot(x='variable', y='value', hue='treatment', data=df,
                 order=['fold1', 'unfold1', 'fold2','unfold2'],
                 gap=.3,
                 linewidth=1.5,
                 boxprops=dict(edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
                 width=0.4)

# plot individual data points
sns.stripplot(x='variable', y='value', hue='treatment',data=df, alpha=0.7, jitter=False)

# add lines for paired data

n=len(df_wide)
for i in range(n):
    plt.plot(['fold1', 'unfold1','fold2', 'unfold2'], [df_wide['fold1'],df_wide['unfold1'],df_wide['fold2'],df_wide['unfold2']],
             color='gray', alpha=0.05, marker='none')

             
# Improve layout
plt.tick_params(axis='both', which='major', labelsize=16)
ax.get_legend().remove()
plt.xlabel('')
plt.ylabel('displacement(um)',fontsize=16)
plt.tight_layout()
plt.show()
#'''