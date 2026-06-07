'''
Please switch to python in cellpose virtual environment
before running this script
'''

from cellpose import models
from cellpose import io
import os
import re
import tifffile

input_folder='D:\\Xiong Lab\\nuclei count\\rois-0928-MMPi\\'
output_folder='D:\\Xiong Lab\\nuclei count\\rois-0928-MMPi\\'
pattern = re.compile(r'([0-9]+)-(e\d+)-([0-9]+)-([a-zA-Z]+)\.tif')
# filename format: date-sample-holdtime-location-nuclei.tif
# eg: 250628-e1-60-leftPSM.tif

# define basic parameters
xres=3.6364 # pixel/um on xy plane
zstep=1 # z step in um
cellDiameter = 21 # cell diameter in pixel
chan = [0,0] # CHANNELS to segement
minVol = 15 # filter small noise
# set model as nuclei
model = models.Cellpose(gpu=True,model_type='nuclei')
zScaleFactor = zstep*xres


for filename in os.listdir(input_folder):
    match = pattern.match(filename)
    if match:
        if os.path.isfile(input_folder+filename.removesuffix('.tif')+'_seg.npy'):
            print(filename+' is already segmented')
            continue
        print('processing '+filename)
        '''
        date=match.group(1)
        sample = match.group(2)
        holdtime = match.group(3)
        location=match.group(4)
        '''
        in_name=input_folder+filename
        out_name=output_folder+filename.removesuffix('.tif')
        with tifffile.TiffFile(in_name) as tif:
            img=tif.asarray()
            masks, flows, styles, diams = model.eval(img, diameter = cellDiameter, 
                                         channels = None,
                                         do_3D = True, 
                                         anisotropy = zScaleFactor, 
                                         min_size = minVol)
            io.masks_flows_to_seg(img, masks, flows, out_name, channels=chan, diams=diams)
            print('finished processing '+filename)