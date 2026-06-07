import os
import re
import numpy as np
import pandas as pd
import body_axis
from matplotlib import pyplot as plt

treatment_info_path="E:\\Xiong Lab\\axis shape data\\treatment_info.csv"
raw_folder="E:\\Xiong Lab\\axis shape data\\"
out_table="summary.csv"
out_img="summary-boxplots.png"

pattern = r"(?P<date>(\d{6}))-(?P<sample>e\d+)-day2\.csv"

os.chdir(raw_folder)
filelist=os.listdir()
results=[]
for filename in filelist:
    match = re.match(pattern, filename)
    if not match:
        continue
    sample = match.group("sample")
    date = match.group("date")
    raw = pd.read_csv(filename)
    x = raw['x'].to_numpy()
    y = raw['y'].to_numpy()
    curvature=body_axis.compute_curvature(x,y)
    summary=body_axis.summarize_feature(curvature)
    summary['date']=date
    summary['sample']=sample
    results.append(summary)

curvature_summary=pd.DataFrame(results)
curvature_summary['date'] = curvature_summary['date'].astype(str)
curvature_summary['sample'] = curvature_summary['sample'].astype(str)

treatment=pd.read_csv(treatment_info_path,encoding='utf-8')
treatment['date'] = treatment['date'].astype(str)
treatment['sample'] = treatment['sample'].astype(str)
merged= pd.merge(curvature_summary, treatment, on=['date', 'sample'], how='inner')

merged.to_csv(out_table, index=False)
body_axis.plot_boxplots(merged,['mean','std','absmax'],out_name=out_img,option='save')