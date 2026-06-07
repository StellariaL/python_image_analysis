import os
import re
import pandas as pd
import roifile

input_folder = "D:\\Xiong Lab\\nuclei count\\"   # folder containing .zip files
output_csv = "D:\\Xiong Lab\\nuclei count\\summary_full.csv"
densities="D:\\Xiong Lab\\nuclei count\\summary.csv"
pattern = re.compile(r'([0-9]+)-(e\d+)-([0-9]+)-rois\.zip')
# filename format: date-sample-holdtime-side-no.tif
# eg: 250628-e1-60-rois.zip

summary_list = []

for filename in os.listdir(input_folder):
    match = pattern.match(filename)
    if match:
        date=match.group(1)
        sample = match.group(2)
        holdtime = match.group(3)
        in_name=input_folder+filename
        rois = roifile.roiread(in_name)
        for roi in rois:
            # ROI name
            roi_name = roi.name
            names=roi_name.split('-')
            x1, y1 = roi.left, roi.top
            x2, y2 = roi.right, roi.bottom
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            row={'date':date,
                 'sample':sample,
                 'holdtime':holdtime,
                 'side':names[0],
                 'no':names[1],
                 'ycoord':cy}
            summary_list.append(row)

summary=pd.DataFrame(summary_list)
densities_df=pd.read_csv(densities)
densities_df['date']=densities_df['date'].apply(str)
densities_df['holdtime']=densities_df['holdtime'].apply(str)
densities_df['no']=densities_df['no'].apply(str)
summary=summary.merge(densities_df,on=['date','sample','holdtime','side','no'])

summary.to_csv(output_csv)