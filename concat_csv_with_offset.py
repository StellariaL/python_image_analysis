import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from utils import bin_profile

folder="E:\\PhD_large_images\\20260512-cellshape\\"
translation={'view1':[0.0,0.0],
             'view2':[2.0,1930.0],
             'view3':[2.0,3859.0],
             'view4':[3.0,5786.0]}
output="E:\\PhD_large_images\\20260512-cellshape\\0603-e1-actin-summary.csv"

pattern = re.compile(r'0603-e1-actin-(view[0-9])-measurement\.csv')
n_bins=10

dfs=[]

for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        view=match.group(1) #group index starts from 1
        offset=translation[view]
        df=pd.read_csv(folder+filename)
        df['X']=df['X']+offset[0]
        df['Y']=df['Y']+offset[1]
        dfs.append(df)

all_data=pd.concat(dfs,ignore_index=True)
all_data.to_csv(output)

cols_to_plot = [c for c in all_data.columns if c not in ['X', 'Y']]

n = len(cols_to_plot)
fig, axes = plt.subplots(
    nrows=n,
    ncols=1,
    figsize=(6, 4*n)
)

for ax, col in zip(axes, cols_to_plot):
    bin_centres,means,sem=bin_profile(df['Y'],df[col],n_bins=n_bins,limits=(0,7986))
    ax.plot(bin_centres,means)
    ax.set_title(f"{col} along A-P")
    ax.relim() # Recalculate data limits based on current data
    ax.autoscale_view() # Update view to match recalculated limits

plt.show()