import roifile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

input_folder="D:/Xiong Lab/20250213-colon/steps/"
output_folder="D:/Xiong Lab/20250213-colon/centroids/"

# Define sample names and ROI file naming pattern
sample_names = ["s2-4"]  # Modify this with your actual sample names
steps = ["step0", "step1", "step2", "step3"]
cmap = cm.get_cmap("hot", len(steps))
colors = [cmap(i) for i in range(len(steps))]


# Function to compute centroid of ROI
def compute_centroid(x_coords, y_coords):
    return np.mean(x_coords), np.mean(y_coords)

# Process each sample
for sample in sample_names:
    image_path=input_folder+f"{sample}-step0.tif"
    outimg_path=output_folder+f"{sample}_centroids.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(6, 6))  # Create a new plot for each sample
    plt.imshow(image, cmap="gray")
    plt.title(f"Centroids for {sample}")
    plt.xlabel("X (right)")
    plt.ylabel("Y (down)")

    # Process each step for the current sample
    for step_idx, step in enumerate(steps):
        roi_zip_path = input_folder+f"{sample}-{step}_rois.zip"  # Construct filename
        if not os.path.exists(roi_zip_path):
            print(f"File not found: {roi_zip_path}")
            continue
        out_path=output_folder+f"{sample}-{step}.csv"
        rois = roifile.roiread(roi_zip_path)
        centroids = []

        for roi in rois:
            if roi.roitype == roifile.ROI_TYPE.FREEHAND:  # Process only freehand ROIs
                x_points, y_points = roi.coordinates()[:, 0], roi.coordinates()[:, 1]
                centroid_x, centroid_y = compute_centroid(x_points, y_points)
                centroids.append((centroid_x, centroid_y))

        # Convert to NumPy array for easy plotting
        centroids = np.array(centroids)
        pd.DataFrame(centroids).to_csv(out_path)

        if len(centroids) > 0:
            plt.scatter(centroids[:, 0], centroids[:, 1], color=colors[step_idx], label=step, alpha=0.7)

    plt.legend()
    plt.savefig(outimg_path)  # Save plot
    plt.show()  # Display plot

print("Processing complete.")