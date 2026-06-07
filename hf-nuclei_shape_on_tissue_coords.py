import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from skimage import io
from scipy.interpolate import splprep, splev

# parameters
sigma=4.5 # number of pixels that skeleton pixels are allowed to deviate from curve
resample=2000 # number of resample points from curve
flip=0 # 1 -> flip 180 deg

folder="E:\\PhD_large_images\\20260116\\"
section="ROCKi-e2-8"

skeleton_path=folder+section+"-skeleton.tif"
nuclei_path=folder+section+"-nuclei_shape.csv"
mask_path=folder+section+"-ectoderm.tif"
imagebg_path=folder+section+"-nuclei.tif"

curvature_path=folder+section+"-curvature.png"
summary_path=folder+section+"-nuclei_shape_summary.png"
csv_path=folder+section+"-nuclei_shape_summary.csv"

# --- Load image ---
img = io.imread(skeleton_path)
background=io.imread(imagebg_path)
background=np.stack([background]*3, axis=-1)
binary = img > 0
coords = np.column_stack(np.where(binary))

# Build graph
G = nx.Graph()
for y, x in coords:
    G.add_node((y, x))

for y, x in coords:
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if (ny, nx_) in G:
                G.add_edge((y, x), (ny, nx_))

# Find endpoints (degree = 1)
endpoints = [n for n in G.nodes if G.degree[n] == 1]
if len(endpoints) != 2:
    raise ValueError("Skeleton is not a simple curve or endpoints not found.")

start, end = endpoints

# Compute ordered path
ordered_pixels = nx.shortest_path(G, source=start, target=end)

# convert to (x,y) array
pts = np.array([(x, y) for y, x in ordered_pixels])
x = pts[:, 0]
y = pts[:, 1]

# calculate cumulative arc length l
dx = np.diff(x)
dy = np.diff(y)
dl = np.sqrt(dx**2 + dy**2)
l = np.concatenate([[0], np.cumsum(dl)])

# fit parametric spline x(l), y(l)
tck, u = splprep([x, y], u=l, s=len(ordered_pixels)*sigma*sigma)

# resample from spline
l_uniform = np.linspace(l.min(), l.max(), resample)
x_l, y_l = splev(l_uniform, tck)

# calculate curvature
dx_l, dy_l = splev(l_uniform, tck, der=1)
ddx_l, ddy_l = splev(l_uniform, tck, der=2)
curvature = (dx_l * ddy_l - dy_l * ddx_l) / (dx_l**2 + dy_l**2)**1.5

# load nuclei centroids
nuclei = pd.read_csv(nuclei_path)
px = nuclei['X'].to_numpy()
py = nuclei['Y'].to_numpy()

# mark ectoderm nuclei
mask=io.imread(mask_path)
mask_in=mask>0
nuclei['if_ectoderm']=mask_in[np.floor(py).astype(int),np.floor(px).astype(int)].astype(int)

# project nuclei onto spline
projected_l = []
for xp, yp in zip(px, py):
    # Compute squared distances to all spline samples
    dist2 = (x_l - xp)**2 + (y_l - yp)**2
    # find closest point
    idx = np.argmin(dist2)
    projected_l.append(l_uniform[idx])

nuclei['projected_l'] = np.array(projected_l)

if flip:
    background=np.flipud(np.fliplr(background))
    ymax=background.shape[0]
    xmax=background.shape[1]
    lmax=l_uniform.max()
    x_l=xmax-x_l
    y_l=ymax-y_l
    curvature=-curvature
    l_uniform=lmax-l_uniform
    nuclei['X']=xmax-nuclei['X']
    nuclei['Y']=ymax-nuclei['Y']
    nuclei['projected_l']=lmax-nuclei['projected_l']

nuclei['Aspect']=nuclei['Major']/nuclei['Minor']
nuclei.to_csv(csv_path)

# plot
fig1,ax=plt.subplots()
ax.imshow(background,cmap='gray', origin='upper')
ax.scatter(x_l,y_l,c=curvature,cmap='bwr',s=4)
plt.savefig(curvature_path,bbox_inches='tight', dpi=300)
plt.close()

ect_nuclei=nuclei[nuclei['if_ectoderm']==1]
mesen_nuclei=nuclei[nuclei['if_ectoderm']==0]
fig2, ax1 = plt.subplots()
ax1.plot(l_uniform,curvature,'k--',label='curvature along tissue')
ax1.set_ylabel('curvature')
ax2 = ax1.twinx()
ax2.plot(ect_nuclei['projected_l'],ect_nuclei['Aspect'],'o',label='ectoderm nuclei aspect ratio')
ax2.plot(mesen_nuclei['projected_l'],mesen_nuclei['Aspect'],'o',label='mesendoderm nuclei aspect ratio')
ax2.set_ylabel('aspect ratio')
fig2.legend()
plt.savefig(summary_path,bbox_inches='tight', dpi=300)
plt.close()