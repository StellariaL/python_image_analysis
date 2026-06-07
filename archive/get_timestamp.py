'''
	1. Get creation time of first loop image (system timestamp)
	2. Assume this is 5s after last loop image is taken, calculate last loop image system timestamp
	3. Read xml, get camera timestamps
	4. Zero camera timestamps
	5. Obtain last loop image camera timestamp
	6. Offset=system timestamp-camera timestamp
	7. Get creation time of all holding images
	8. Timestamp=system timestamp-offset
    9. Write timestamps as csv (length=loop+hold)
'''

import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import re

home_folder="D:/Xiong Lab/20260127/"

pattern = re.compile(r'([a-zA-Z]+)(\d+)-hold_image_stack\.tif')

interval=60 # assume loop images are saved 30s after the last one being taken

for filename in os.listdir(home_folder):
	match = pattern.match(filename)
	if match:
		treatment=match.group(1) #group index starts from 1
		no = match.group(2)
		samplename=treatment+no
		print('Processing sample '+samplename)
		hold_folder=home_folder+samplename+"-hold/"
		loop_ref=home_folder+samplename+"-load_image_files/0001.tif"
		xiseq_path=home_folder+samplename+"-load_image.xiseq"
		output_file=home_folder+samplename+"-time.csv"
		hold_ts=[]
		loop_ts=[]

		last_loop_sys=os.path.getmtime(loop_ref)-interval

		tree = ET.parse(xiseq_path)
		root = tree.getroot()
		for file in root.findall('file'):
			cam_ts=float(file.attrib['timestamp'])
			loop_ts.append(cam_ts/(10**6))

		loop_ts=np.array(loop_ts)
		loop_ts=np.sort(loop_ts)
		loop_ts=loop_ts-loop_ts[0]
		last_loop_cam=loop_ts[-1]
		offset=last_loop_sys-last_loop_cam

		for filename in os.listdir(hold_folder):
			if filename.endswith('.tif'):
				filepath=hold_folder+filename
				ts=os.path.getmtime(filepath)-offset
				hold_ts.append(ts)
		hold_ts=np.array(hold_ts)
		hold_ts=np.sort(hold_ts)

		timestamps=pd.DataFrame(np.concatenate((loop_ts,hold_ts)))
		timestamps.to_csv(output_file)