import pandas as pd
import matplotlib.pyplot as plt
from utils import bin_profile

data=pd.read_csv("E:\\PhD_large_images\\20260512-cellshape\\0603-e1-actin-summary.csv")

n_bins=100

bin_centres,means,sem=bin_profile(data['Y'],data['Area'],n_bins=n_bins,limits=(0,7986))

plt.plot(bin_centres,means)
plt.show()