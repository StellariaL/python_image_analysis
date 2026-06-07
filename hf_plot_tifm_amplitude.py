import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

filename="E:\\PhD_large_images\\20260517-TiFM_sin_stiffness\\fit_results.csv"

df=pd.read_csv(filename)


# Set the visual style
sns.set_theme(style='white',font_scale=1.5)

# Create boxplot
plt.figure(figsize=(6, 4))
ax = sns.boxplot(x='treatment', y='AR/Ac', data=df,
                 order=['air', 'ctrl', 'ROCKi'],
                 linewidth=1.5,
                 boxprops=dict(facecolor='white', edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
                 width=0.4)

# plot individual data points
sns.stripplot(x='treatment', y='AR/Ac', data=df, color='black', alpha=0.7, jitter=False)


# Improve layout
#plt.ylim(-0.1,0.5)
plt.title('')
plt.tight_layout()
plt.show()
