import re
import numpy as np
import pandas as pd
import tifffile

roic_folder="D:\\Xiong Lab\\20250913-sora\\"
stack_folder="D:\\Xiong Lab\\20250913-sora\\"
roi_folder="D:\\Xiong Lab\\filaments\\rois-fib\\"
proj_folder="D:\\Xiong Lab\\filaments\\maxprojections\\"

roi_h=400
roi_w=200 #in pixel
channel=2 #channel to keep; index starts from 1
channelname="fib" #for naming files
zrange=[1,25] # z slices to keep (inclusive); index starts from 1. To keep all slices, set as empty.

pattern = re.compile(r'([0-9]+)-(e\d+)-([0-9]+)-roictr\.csv')
# filename format: date-sample-holdtime-roictr.csv
# eg: 250628-e1-60-roictr.csv
channelid=channel-1
'''
for filename in os.listdir(roic_folder):
    match = pattern.match(filename)
    if match:
        sample=filename.removesuffix('-roictr.csv')
        print("processing "+sample)
        roic_path=roic_folder+filename
        img_path=stack_folder+sample+'.tif'
        roi_c=pd.read_csv(roic_path,header=0)
        img=tifffile.imread(img_path)

        for index,row in roi_c.iterrows():
            x,y=int(row['x']),int(row['y'])
            x1, x2 = x - roi_w//2, x + roi_w//2
            y1, y2 = y - roi_h//2, y + roi_h//2
            crop=img[:,0,y1:y2,x1:x2] # only keep channel 1 (nuclei)
            out_path=roi_folder+sample+'-'+str(index+1)+'.tif'
            tifffile.imwrite(out_path,crop)
            '''
filename="250913-e2-60-roictr.csv"
match = pattern.match(filename)
if match:
    sample=filename.removesuffix('-roictr.csv')
    print("processing "+sample)
    roic_path=roic_folder+filename
    img_path=stack_folder+sample+'.tif'
    roi_c=pd.read_csv(roic_path,header=0)
    img=tifffile.imread(img_path)
    if zrange==[]:
        slices=range(img.shape[0])
    else:
        slices=range(zrange[0]-1,zrange[1])
    for index,row in roi_c.iterrows():
        x,y=int(row['x']),int(row['y'])
        x1, x2 = x - roi_w//2, x + roi_w//2
        y1, y2 = y - roi_h//2, y + roi_h//2
        crop=img[slices,channelid,y1:y2,x1:x2] # only keep channel 1 (nuclei)
        roi_path=roi_folder+"-".join([sample,str(index+1)])+".tif"
        tifffile.imwrite(roi_path,crop)

        maxproj=np.amax(crop,axis=0)
        proj_path=proj_folder+"-".join([sample,str(index+1),channelname,"maxproj"])+".tif"
        tifffile.imwrite(proj_path,maxproj)