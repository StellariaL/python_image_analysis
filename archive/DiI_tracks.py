import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

xmlpath="D:\\Xiong Lab\\20250508\\_Tracks.xml"

# Load and parse the XML file
tree = ET.parse(xmlpath)  # Replace with your XML file path
root = tree.getroot()

# List to store parsed data
data = []

# Iterate through each particle
for i, particle in enumerate(root.findall('particle')):
    particle_id = i+1
    for detection in particle.findall('detection'):
        t = float(detection.attrib['t'])
        x = float(detection.attrib['x'])
        y = float(detection.attrib['y'])
        data.append({
            't': t,
            'x': x,
            'y': y,
            'particle': particle_id
        })

# Create a DataFrame
df = pd.DataFrame(data)
initial_pos=df.loc[df['t']<=5,['x','y','particle']].groupby('particle').mean().rename(columns={"x": "x_0", "y": "y_0"})
end_pos=df.loc[df['t']>=168,['x','y','particle']].groupby('particle').mean().rename(columns={"x": "x_45", "y": "y_45"})
merged=pd.merge(initial_pos,end_pos,how='inner',on='particle')
merged['u']=merged['x_45']-merged['x_0']
merged['v']=merged['y_45']-merged['y_0']
merged['disp']=np.sqrt(merged['u']**2+merged['v']**2)
merged.to_csv('D:\\Xiong Lab\\20250508\\dii_displacements.csv',index=False)
'''
fig, axes = plt.subplots(2,3)
axes=axes.flatten()
norm = plt.Normalize(vmin=merged['disp'].min(), vmax=merged['disp'].max())
cmap = plt.get_cmap('inferno')
for index,row in merged.iterrows():
    color = cmap(norm(row['disp']))
    axes[0].arrow(row['x_0'],row['y_0'],row['u'],row['v'],
             head_width=2, head_length=4,
             fc=color, ec=color,
             alpha=0.7, linewidth=1.5)
axes[0].yaxis.set_inverted(True)
axes[1].scatter(merged['disp'],merged['y_0'])
axes[1].yaxis.set_inverted(True)
axes[2].scatter(-merged['u'],merged['y_0'])
axes[2].yaxis.set_inverted(True)
axes[3].scatter(merged['x_0'],merged['v'])
axes[4].scatter(merged['v'],merged['y_0'])
axes[4].yaxis.set_inverted(True)
axes[5].scatter(merged['x_0'],merged['y_0'])
axes[5].yaxis.set_inverted(True)
for ax in axes:
    ax.set_aspect('equal')
plt.show()
'''