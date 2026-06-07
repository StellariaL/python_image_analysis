import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("E:\\PhD_large_images\\20260528-dispase\\nc_length.csv")

# Prepare figure
plt.figure(figsize=(10, 6))

# Unique samples
samples = df['sample'].unique()

# Marker mapping by treatment
marker_map = {
    'ctrl': 'o',
    'dispase': 'x'
}

# Loop through samples and plot
for sample in samples:

    sub = df[df['sample'] == sample]
    treatment = sub['treatment'].iloc[0]   # same for all rows of this sample
    marker = marker_map[treatment]
    '''
    plt.plot(
        sub['time'],
        sub['nc_length'],
        marker=marker,
        label=sample,
        linewidth=1.5
    )
    '''
    sub = sub.sort_values("time")

    # Compute growth rate
    dt = sub["time"].diff()
    dl = sub["nc_length"].diff()
    rate = dl / dt

    # Midpoints for plotting
    mid_time = (sub["time"][:-1].values + sub["time"][1:].values) / 2

    plt.plot(
        mid_time,
        rate[1:],   # skip first NaN
        marker=marker,
        label=sample,
        linewidth=1.5
    )

plt.xlabel("Time")
plt.ylabel("nc_length")
plt.title("nc_length vs Time for Each Sample")
plt.legend(title="Sample", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
