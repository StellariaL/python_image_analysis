'''
Template for iterating files in a folder, with re split of filename
'''

import os
import re

folder=""

pattern = re.compile(r'([0-9]+)-(e\d+)-commonpart\.suffix')

for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        firstpart=match.group(1) #group index starts from 1
        secondpart = match.group(2)