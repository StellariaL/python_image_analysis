import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import ticker

# Example data
values = [0.16113722,0.025602658,0.498893146,0.351800484,0.544476869,0.235709592,0.354406677,-0.000637461,0.479841658,0.284632856,0.252499986,0.541766128,0.320989862,0.599083319]
labels = [15,15,15,30,30,30,30,5,5,5,5,60,60,60]

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'Value': values,
    'Condition': labels
})

# Set the visual style
sns.set_theme(style='white')

# Create the plot
plt.figure(figsize=(6, 4))
ax = sns.boxplot(x='Condition', y='Value', data=df, linewidth=1.5,  # Thicker lines for visibility
                 boxprops=dict(facecolor='white', edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),
                 width=0.4)
sns.stripplot(x='Condition', y='Value', data=df, color='black', alpha=0.7, jitter=True)

# Improve layout
plt.ylabel('Residual displacement',fontsize=18)
plt.xlabel('Holding time (min)',fontsize=18)
plt.ylim([0,1])
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.show()
