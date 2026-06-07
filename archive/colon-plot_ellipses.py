import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse

# Load the dataset
file_path = "D:\\Xiong Lab\\20250312-colon\\Results.csv"
df = pd.read_csv(file_path,header=0)

# Extract sample and step information
df[['Sample', 'Step']] = df['Name'].str.extract(r'(c\d\.*\d*)-step(\d+)')
df['Step'] = df['Step'].astype(int)

print(df.head())

stretch_samples = df[df["type"] == "stretch"]
compress_samples = df[df["type"] == "compress"]

samples = df["Sample"].unique()
num_samples = len(samples)

def plot_ellipses(sample_data, title):
    positive_samples = sample_data[sample_data["status"] == "+"]["Sample"].unique()
    negative_samples = sample_data[sample_data["status"] == "-"]["Sample"].unique()
    
    num_cols = max(len(positive_samples), len(negative_samples))
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), sharex=True, sharey=True)
    
    if num_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    colors = plt.cm.inferno(np.linspace(0, 1, df["Step"].nunique()))
    
    for row_idx, samples in enumerate([positive_samples, negative_samples]):
        for col_idx, sample in enumerate(samples):
            ax = axes[row_idx, col_idx]
            sample_data_subset = sample_data[sample_data["Sample"] == sample]
            
            for i, (step, step_data) in enumerate(sample_data_subset.groupby("Step")):
                for _, row in step_data.iterrows():
                    major=row["Major"]
                    minor=row["Minor"]
                    ellipse = Ellipse(
                        (row["X"],row["Y"]), row["Major"], row["Minor"],
                        angle=row["Angle"], edgecolor=colors[i], facecolor='none', linewidth=1
                    )
                    ax.add_patch(ellipse)
            ax.set_xlim(500,1500)
            ax.set_ylim(500,1500)
            ax.set_title(sample)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_ellipses(stretch_samples, "Stretch Samples")
plot_ellipses(compress_samples, "Compress Samples")
