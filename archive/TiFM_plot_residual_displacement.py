import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data
values = [0.0057,0.0137,-0.0506,0.2476,0.1984,0.1395,-0.0188,0.1273,-0.0719,0.2601]
labels = ['load','load','load','unload','unload','unload','load','unload','load','unload']

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
plt.ylim(-0.1,0.5)
plt.tight_layout()
plt.show()
