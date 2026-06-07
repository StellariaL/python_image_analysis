import os
import re
import pandas as pd
import numpy as np

input_folder='D:\\Xiong Lab\\nuclei count\\rois-0928-MMPi\\'
output_folder='D:\\Xiong Lab\\nuclei count\\rois-0928-MMPi\\'
pattern = re.compile(r'([0-9]+)-(e\d+)-([0-9]+)-([a-zA-Z]+)_seg\.npy')
# filename format: date-sample-holdtime-location_seg.npy
# eg: 250628-e1-60-left_seg.npy

summary_list=[]

for filename in os.listdir(input_folder):
    match = pattern.match(filename)
    if match:
        date=match.group(1)
        sample = match.group(2)
        holdtime = match.group(3)
        location=match.group(4)
        data = np.load(input_folder+filename, allow_pickle=True).item()
        masks = data["masks"]
        num=str(masks.max())
        row={'date':date,
                'sample':sample,
                'holdtime':holdtime,
                'location':location,
                'counts':num}
        summary_list.append(row)

summary = pd.DataFrame(summary_list)
summary.to_csv(output_folder+'summary.csv')