import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

input_folder="D:/Xiong Lab/20250213-colon/"
output_folder="D:/Xiong Lab/20250213-colon/centroids/"

# User-defined parameters
sample_names = ["s2-4"]  # Modify with your sample name
step_a = "step0"  # Define the first step
step_b = "step2"  # Define the second step
distance_threshold = 40  # Define the max allowed displacement to avoid mismatches
colormap = "hot"  # Choose a matplotlib colormap for vector coloring

def load_centroids_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return np.array([])

    df = pd.read_csv(csv_path)  # Load CSV file
    if df.shape[1] >= 3:  # Ensure at least three columns exist
        return df.iloc[:, 1:3].values  # Extract x, y coordinates (ignoring index column)
    else:
        print(f"Invalid format in {csv_path}, skipping.")
        return np.array([])

def find_nearest_neighbors(centroids_a, centroids_b, threshold):
    matched_vectors = []
    displacement_magnitudes = []
    
    for c_a in centroids_a:
        distances = np.linalg.norm(centroids_b - c_a, axis=1)  # Compute Euclidean distances
        min_idx = np.argmin(distances)  # Find nearest neighbor
        min_dist = distances[min_idx]

        if min_dist <= threshold:  # Apply threshold condition
            matched_vectors.append((c_a, centroids_b[min_idx]))
            displacement_magnitudes.append(min_dist)

    return matched_vectors, displacement_magnitudes


for sample in sample_names:

    image_path=input_folder+f"steps/{sample}-step0.tif"
    outimg_path=output_folder+f"{sample}_{step_a}_to_{step_b}.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fig,ax=plt.subplots(figsize=(6, 6))  # Create a new plot for each sample
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Vector Map: {step_a} to {step_b}")
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (down)")

    # Load centroids for the two selected steps from CSV
    centroids_a = load_centroids_from_csv(input_folder+f"centroids/{sample}-{step_a}.csv")
    centroids_b = load_centroids_from_csv(input_folder+f"centroids/{sample}-{step_b}.csv")

    # Find matched centroids
    if centroids_a.size > 0 and centroids_b.size > 0:
        matched_vectors, displacement_magnitudes = find_nearest_neighbors(centroids_a, centroids_b, distance_threshold)
    else:
        matched_vectors, displacement_magnitudes = [], []

    if displacement_magnitudes:
        displacement_magnitudes = np.array(displacement_magnitudes)
        norm = plt.Normalize(vmin=displacement_magnitudes.min(), vmax=displacement_magnitudes.max())
        cmap = plt.get_cmap(colormap)
    else:
        norm = None
        cmap = None

    # Plot vectors
    for i, (start, end) in enumerate(matched_vectors):
        color = cmap(norm(displacement_magnitudes[i])) if cmap else "red"
        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], 
                head_width=2, head_length=4, fc=color, ec=color, alpha=0.7, linewidth=1.5)

# Add colorbar
    if norm:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)  # Explicitly assign axes
        cbar.set_label("Displacement Magnitude (pixels)")


    plt.savefig(outimg_path)
    plt.show()

    print("Vector map generation complete.")
