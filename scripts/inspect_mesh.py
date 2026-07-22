import trimesh
import numpy as np
import os
p = os.path.join('exp','1999JV6','meshes','mesh_001000.ply')
mesh = trimesh.load(p)
verts = np.asarray(mesh.vertices)
print('path:', p)
print('n_verts:', len(verts))
print('min:', np.min(verts, axis=0))
print('max:', np.max(verts, axis=0))
print('extent:', np.max(verts, axis=0) - np.min(verts, axis=0))
print('center:', np.mean(verts, axis=0))
print('radius:', np.max(verts, axis=0) - np.mean(verts, axis=0))

p = os.path.join('public_data','1999JV6','1999 JV6 Radar_60.ply')
mesh = trimesh.load(p)
verts = np.asarray(mesh.vertices)
print('path:', p)
print('n_verts:', len(verts))
print('min:', np.min(verts, axis=0))
print('max:', np.max(verts, axis=0))
print('extent:', np.max(verts, axis=0) - np.min(verts, axis=0))
print('center:', np.mean(verts, axis=0))
