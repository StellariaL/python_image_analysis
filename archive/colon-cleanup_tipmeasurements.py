import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path_pref="D://Xiong Lab//20250312-colon//tippos-"
condition="compress"

file_path = file_path_pref+condition+".xlsx"
out_path=file_path_pref+condition+"_rearranged.xlsx"

df = pd.read_excel(file_path)

transformed_data = []

    # Group by sample and location
for (sample, location), group in df.groupby(["sample", "location"]):
    group = group.sort_values("no").reset_index(drop=True)
    
    # Each image has 2 pairs of (x_left, x_right), 4 images in total
    num_images = len(group) // 2  # Should be 4 images per sample-location
    
    status=group.loc[1,"status"]

    for i in range(num_images):
        x_left = group.loc[2 * i, "x"]
        x_right = group.loc[2 * i + 1, "x"]
        transformed_data.append([sample, location, i + 1, x_left, x_right,status])

# Convert to DataFrame
transformed_df = pd.DataFrame(transformed_data, columns=["sample", "location", "image", "x_left", "x_right","status"])

transformed_df.to_excel(out_path, index=False)

aligned_data = transformed_df.copy()
aligned_data["distance"]=aligned_data["x_right"]-aligned_data["x_left"]
aligned_data["distance_diff"]=aligned_data["distance"]
for (sample, location), group in aligned_data.groupby(["sample", "location"]):
    x_left_ref = group.loc[group["image"] == 1, "x_left"].values[0]
    dist_ref=group.loc[group["image"] == 1, "distance"].values[0]
    aligned_data.loc[(aligned_data["sample"] == sample) & (aligned_data["location"] == location), ["x_left", "x_right"]] -= x_left_ref
    aligned_data.loc[(aligned_data["sample"] == sample) & (aligned_data["location"] == location), ["distance_diff"]] -= dist_ref


fig,axes=plt.subplots(1,3,sharex=True,sharey=True)
sns.set_style("whitegrid")

for (sample, location), group in aligned_data.groupby(["sample", "location"]):
    s=group.iloc[0]
    if s["status"]=="+":
        color="red"
    else:
        color="blue"
    axes[0].plot(group["image"], group["distance_diff"], color=color, linestyle="-", label=f"Sample {sample}, Location {location}")
    axes[1].plot(group["image"], group["x_left"], color=color, linestyle="-", label=f"Sample {sample}, Location {location} (Left)")
    axes[2].plot(group["image"], group["x_right"], color=color, linestyle="-", label=f"Sample {sample}, Location {location} (Right)")

axes[0].set_title("distance between tips")
axes[1].set_title("left tip")
axes[2].set_title("right tip")

plt.suptitle("Aligned tip Positions")
plt.legend(loc="best", fontsize="small", ncol=2)
plt.show()