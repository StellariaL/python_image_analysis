'''
This script builds a triangular surface mesh of an object in a 3D stack.

workflow:
1. read stack
2. apply a 3D Gaussian blur (separate sigma for z and xy)
3. run marching cubes on the blurred stack at an intensity level taken from a
   percentile of the blurred intensities
4. save the mesh as .npz (for python) and .ply (for MeshLab / Blender / Fiji)
5. open the mesh in napari, overlaid on the stack it came from

parameters:
    sigma_xy, sigma_z: Gaussian blur sigma (px) in xy and in z
    level_percentile: percentile of blurred intensity used as the iso-surface level
    pixel_size, z_step: voxel size (um); mesh coordinates come out in um
    mesh_step: marching cubes step size (px); >1 gives a coarser, lighter mesh

outputs:
    *-mesh.npz: verts (N,3) as (z,y,x) in um, faces (M,3), normals (N,3), values (N,)
    *-mesh.ply: same mesh as (x,y,z), for external viewers

tips:
Reload with d = np.load(path); verts, faces = d['verts'], d['faces'].
Blur heavily enough that the surface is smooth; marching cubes follows noise.
The stack layer starts hidden; toggle its eye icon to check the surface against
the data, and use the napari camera button to switch between 2D and 3D views.
'''

import napari
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

# === Input ===
tiff_path = "D:\\Ruoheng_Li\\20260818-actin-10x-live\\10x-actin-d-live-t49.tif"

# parameters
sigma_xy = 20
sigma_z = 2
level_percentile = 50
pixel_size = 0.6745408
z_step = 2.5
mesh_step = 2

mesh_path = tiff_path.removesuffix('.tif') + '-mesh.npz'
ply_path = tiff_path.removesuffix('.tif') + '-mesh.ply'


def write_ply(path: str, verts: np.ndarray, faces: np.ndarray,
              normals: np.ndarray) -> None:
    """Write a binary little-endian PLY, converting (z, y, x) to (x, y, z)."""
    vertex_data = np.empty(len(verts), dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
    ])
    vertex_data['x'], vertex_data['y'], vertex_data['z'] = verts.T[::-1]
    vertex_data['nx'], vertex_data['ny'], vertex_data['nz'] = normals.T[::-1]

    face_data = np.empty(len(faces), dtype=[('n', 'u1'), ('v', '<i4', (3,))])
    face_data['n'] = 3
    face_data['v'] = faces

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertex_data.tobytes())
        f.write(face_data.tobytes())


stack = tiff.imread(tiff_path)
if stack.ndim != 3:
    raise ValueError("Input image must be a grayscale z-stack (Z, Y, X).")

blurred = gaussian_filter(stack.astype(np.float32), sigma=(sigma_z, sigma_xy, sigma_xy))

level = float(np.percentile(blurred, level_percentile))
if not blurred.min() < level < blurred.max():
    raise ValueError(f"Level {level} is outside the blurred intensity range; "
                     "adjust level_percentile.")

# verts are (z, y, x) in um because of spacing
verts, faces, normals, values = marching_cubes(
    blurred,
    level=200,
    spacing=(z_step, pixel_size, pixel_size),
    step_size=mesh_step,
)
print(f"level {level:.1f} -> {len(verts)} vertices, {len(faces)} faces")

np.savez_compressed(
    mesh_path,
    verts=verts,
    faces=faces,
    normals=normals,
    values=values,
    level=level,
    spacing=np.array([z_step, pixel_size, pixel_size]),
)
write_ply(ply_path, verts, faces, normals)

# napari works in (z, y, x), so verts go in as they come out of marching cubes;
# scaling the stack by the voxel size puts both layers in the same um space
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(
    stack,
    name='stack',
    scale=(z_step, pixel_size, pixel_size),
    colormap='gray',
    rendering='attenuated_mip',
    visible=False,
)
viewer.add_surface(
    (verts, faces),
    name='surface',
    shading='smooth',
)

napari.run()
