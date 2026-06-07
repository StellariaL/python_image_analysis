import os
import re
import numpy as np
import pandas as pd
import body_axis
from matplotlib import pyplot as plt

ref_length=50 # the point this many pixel away from the endpoint will be aligned to the x axis.
ref_ind=int(-ref_length-1)

treatment_info_path="E:\\Xiong Lab\\axis shape data\\TiFM_bending\\info.csv"
raw_folder="E:\\Xiong Lab\\axis shape data\\TiFM_bending\\"
out_table="summary.csv"
out_img="summary-boxplots.png"

treatment_info = pd.read_csv(treatment_info_path)

# Separate treatments
ctrl_info = treatment_info[treatment_info['treatment'] == 'ctrl']
pd_info = treatment_info[treatment_info['treatment'] == 'PD']




# Initialize subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
axes[0].set_title('4ctrl')
axes[1].set_title('4PD')

for ax, info, title in zip(axes, [ctrl_info, pd_info], ['4ctrl', '4PD']):
    for _, row in info.iterrows():
        filename = f"{row['date']}-{row['sample']}-day2.csv"
        filepath = os.path.join(raw_folder, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        try:
            raw = pd.read_csv(filepath)
            x = raw['x'].to_numpy()
            y = raw['y'].to_numpy()
            x_aligned = x - x[-1]
            y_aligned = y - y[-1]
            if len(x) <= ref_length:
                raise ValueError(f"The curve {filepath} does not have enough points for alignment.")
            dx = x_aligned[ref_ind] - x_aligned[-1]
            dy = y_aligned[ref_ind] - y_aligned[-1]
            angle = -np.arctan2(dy, dx)
            x_rotated = x_aligned * np.cos(angle) - y_aligned * np.sin(angle)
            y_rotated = x_aligned * np.sin(angle) + y_aligned * np.cos(angle)
            ax.plot(x_rotated, y_rotated)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

plt.suptitle('Aligned Notochord Shapes')
plt.tight_layout()
plt.show()

