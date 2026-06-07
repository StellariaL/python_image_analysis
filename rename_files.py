import os
import re

timepoint_map = {
    "before": "0",
    "0h": "1",
    "1h": "2",
    "2h": "3",
    "3h":"4",
    "4h": "5",
    "5h":"6",
    "6h": "7"
}

pattern = re.compile(r'([a-zA-Z0-9]+)-e(\d+)-([a-zA-Z0-9]+)')

# Set the path to your folder
in_path ="D:\\Xiong Lab\\force-displacement\\FGFi-final\\good\\PD\\"
out_path="D:\\Xiong Lab\\force-displacement\\FGFi-final\\good\\"


# Loop through all files in the folder
for filename in os.listdir(in_path):
    match = pattern.match(filename)
    if match:
        date = match.group(1)
        no = match.group(2)
        type=match.group(3)
        new_filename=f"{date}-PD{no}-{type}"
        old_path=os.path.join(in_path,filename)
        new_path = os.path.join(out_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
'''
for filename in os.listdir(in_path):
    # Check if file is a .zip and starts with 'MAX_'
    if filename.endswith('.xlsx'):
        # New filename with 'MAX_' removed
        new_filename = '5-'+filename
        
        # Full paths
        old_path = os.path.join(in_path, filename)
        new_path = os.path.join(out_path, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

for filename in os.listdir(folder_path):
    # Check if file is a .zip and starts with 'MAX_'
    if filename.endswith('_image_stack-c.tif'):
        # New filename with 'MAX_' removed
        new_filename = filename.removesuffix('-c.tif')+'.tif'
        
        # Full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
'''
print("Done.")
