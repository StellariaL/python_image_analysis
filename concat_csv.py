import pandas as pd
import glob
import os

folder = "D:\\Ruoheng_Li\\260808\\"

files = glob.glob(os.path.join(folder, "*_xy.csv"))

df = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True
)

df.to_csv(os.path.join(folder, "measurements.csv"), index=False)