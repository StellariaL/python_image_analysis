import os
import re
import shutil
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# Define parameters
ROI = (0, 867, 2048, 192)  # (x_start, y_start, width, height)
INPUT_DIR = r"\\blue.cam.ac.uk\\RFS\\rfs-xiong-8CLlgDpM7KI\\Ruoheng Li\\TiFM\\20250312-colon"  # Folder containing subfolders of images
THRESHOLD = 250  # Adjust based on your data
OUTPUT_DIR = "D:\\Xiong Lab\\20250312-colon"  # Output folder for selected images
LOG_FILE = os.path.join(OUTPUT_DIR, "selection_log.txt")


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex pattern to match filenames
pattern = re.compile(r"a(?:\((\d+)\))?\.tif")

with open(LOG_FILE, "w") as log:
    for folder in ['s1-c5-compress','s1-c6-compress']:
        folder_path = os.path.join(INPUT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Get and sort image filenames based on numeric order
        images = []
        for filename in os.listdir(folder_path):
            match = pattern.match(filename)
            if match:
                index = int(match.group(1)) if match.group(1) else 0
                images.append((index, filename))
        images.sort()
        
        prev_array = None
        rmsd_values = []
        image_numbers = []
        selected_images=[]
        step_no=0
        prev_filepath=None
        
        for i, filename in images:
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "rb") as f:
                image = tiff.imread(f)
            
            # Extract ROI and sum along height direction
            x, y, w, h = ROI
            roi_data = image[y:y+h, x:x+w]
            intensity_profile = np.sum(roi_data, axis=0)
            
            if prev_array is not None:
                rmsd = np.sqrt(np.mean((intensity_profile - prev_array) ** 2))
                rmsd_values.append(rmsd)
                image_numbers.append(i-1)
                if rmsd > THRESHOLD:
                    new_filename = f"{folder}-step{step_no}.tif"
                    shutil.copy(prev_filepath, os.path.join(OUTPUT_DIR, new_filename))
                    selected_images.append(str(i-1))
                    step_no+=1
            
            prev_array = intensity_profile
            prev_filepath=filepath
        
        # Plot RMSD values
        if rmsd_values:
            plt.figure()
            plt.plot(image_numbers, rmsd_values, marker='o', linestyle='-')
            plt.xlabel("Image Number")
            plt.ylabel("Root Mean Square Difference")
            plt.title(f"RMSD Plot for {folder}")
            plt.savefig(os.path.join(OUTPUT_DIR, f"{folder}_rmsd_plot.png"))
        if selected_images:
            log.write(f"{folder}: {', '.join(selected_images)}\n")
        print("finished processing:"+folder)