import os
import re
import numpy as np
import pandas as pd
import tifffile

input_folder='D:\\Xiong Lab\\nuclei count\\0928-MMPi\\'
output_folder='D:\\Xiong Lab\\nuclei count\\0928-MMPi\\tiles\\'
pattern = re.compile(r'([0-9]+)-(e\d+)-([0-9]+)-([a-zA-Z]+)-nuclei\.tif')

tilen=5 # number of tiles
overlap=0.1 # proportion of overlap 

for filename in os.listdir(input_folder):
    match = pattern.match(filename)
    if match:
        print('processing '+filename)
        in_name=input_folder+filename
        with tifffile.TiffFile(in_name) as tif:
            img=tif.asarray()
            h=img.shape[1]
            tileh=int(h/((1-overlap)*(tilen-1)+1))
            tilestart=np.arange(tilen)*int(tileh*(1-overlap))
            tileend=tilestart+tileh
            tileend[-1]=h-1
            config=pd.DataFrame(tilestart)
            config.to_csv(output_folder+filename.removesuffix('.tif')+'-tiles.csv')
            for i in range(tilen):
                tile=img[:,tilestart[i]:tileend[i],:]
                tile_name=output_folder+filename.removesuffix('.tif')+'-'+str(i+1)+'.tif'
                tifffile.imwrite(tile_name,tile)