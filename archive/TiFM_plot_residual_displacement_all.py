import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Example data: each array has N values for N groups
measurement1 = [1.5786,3.0173,1.9233,-0.0063,4.6401] #init
measurement2 = [84.0158, 95.3827, 109.1077, 74.0101, 91.7461]#max
measurement3 = [20.8055,18.9284,15.2252,9.4216,23.8600]#end

# Number of groups
n_groups = len(measurement1)

# Combine into a DataFrame for plotting
df = pd.DataFrame({
    'Group': np.repeat(np.arange(n_groups), 3),
    'Measurement': ['1'] * n_groups + ['2'] * n_groups + ['3'] * n_groups,
    'Value': measurement1 + measurement2 + measurement3
})

# Set style
sns.set_theme(style='white')

# Create the figure
plt.figure(figsize=(6, 4))

# Boxplot
ax = sns.boxplot(x='Measurement', y='Value', data=df,
                 linewidth=1.5,  # Thicker lines for visibility
                 boxprops=dict(facecolor='white', edgecolor='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(markerfacecolor='black', markeredgecolor='black'),width=0.4)


# Overlay individual trajectories
for i in range(n_groups):
    plt.plot(['1', '2', '3'], [measurement1[i], measurement2[i], measurement3[i]],
             color='gray', alpha=0.6, marker='o')

# Title and layout
plt.title('Measurements Across Conditions with Group Trajectories')
plt.tight_layout()
plt.show()
