import numpy as np
import tifffile
import matplotlib.pyplot as plt

# --- Configuration ---

# Path to your TIFF stack
tiff_path = "D:/Xiong Lab/20250628-confocal/TiFM/e1.tif"

roi_h=200
roi_w=200

roi_positions = [
    (1572,1128),
    (846,1152)
    # Add more as needed
]

# --- Load TIFF stack ---

# The image should have shape (z, y, x, channel)
# Adjust this if the TIFF stack has different dimension order
img = tifffile.imread(tiff_path)

# Check shape
print(f"Loaded stack shape: {img.shape}")  # Expected: (z, y, x, c)

# Ensure it's in (z, y, x, c) format
if img.ndim == 4:
    z_slices, n_channels, height, width = img.shape
else:
    raise ValueError("Expected a 4D stack with shape (z, y, x, channels)")


all_profiles = []

for roi_index, (x1, y1) in enumerate(roi_positions):
    # Clip ROI boundaries to avoid out-of-bounds access
    x2 = min(x1 + roi_w, width)
    y2 = min(y1 + roi_h, height)
    x1_clipped = max(0, x1)
    y1_clipped = max(0, y1)

    # Store the profile for this ROI
    profile = []

    for z in range(z_slices):
        roi = img[z, 2, y1_clipped:y2, x1_clipped:x2]  # Channel 3 = index 2
        mean_intensity = roi.mean()
        profile.append(mean_intensity)

    all_profiles.append(profile)

# --- Plotting ---

plt.figure(figsize=(8, 5))

for i, profile in enumerate(all_profiles):
    plt.plot(range(z_slices), profile, marker='o')

plt.xlabel('Z Slice')
plt.ylabel('Mean Intensity (Channel 3)')
plt.title('Z-Profiles of Multiple ROIs')
plt.legend(['right','left'])
plt.grid(True)
plt.tight_layout()
plt.show()