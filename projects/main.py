from utils import *
from mesh import *
from deformation import *
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def modifier(point):
    x = point[0] + 4
    y = point[1] + 4
    z = 0.5 * (x ** 2.0) + 3.0
    return [x, y, z]


def modifier_debug(point):
    x = point[0]
    y = point[1]
    z = point[2] + 2.0
    return [x, y, z]

# --------------------------------------------------------
# Generate synthetic data
# --------------------------------------------------------

# Generate simple surface for test
gen = MeshGenerator()
surf, tri = gen.make_plane(nx=20, ny=5, dx=1, dy=1)
surf[:, 0] -= 9.5
surf[:, 1] -= 2.5
surf[:, 2] += 1.0
surf_t = gen.transform(surf, modifier)
N = surf.shape[0]

# Display the problem
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(surf[:,0], surf[:,1], surf[:,2], triangles=tri)
ax.plot_trisurf(surf_t[:,0], surf_t[:,1], surf_t[:,2], triangles=tri)
plt.show(block=False)

# --------------------------------------------------------
# Generate Laplacian
# --------------------------------------------------------
mesh_src = Mesh(surf, tri)
mesh_tgt = Mesh(surf_t, tri)
#Deg, Adj, Lap = mesh_src.compute_laplacian('normalized')
Deg, Adj, Lap = mesh_src.compute_laplacian('combinatorial')

L = np.array(Lap.todense())


# Sanity check
assert (Adj - Adj.transpose()).count_nonzero() == 0
assert (Lap - Lap.transpose()).count_nonzero() == 0 #.count_nonzero() == 0


# --------------------------------------------------------
# Laplacian mesh deformation
# --------------------------------------------------------
# Define anchor's indexes

"""
minx = min(surf[:, 0])
maxx = max(surf[:, 0])
ancIdx = []
ancIdx.extend(np.where(surf[:, 0] == minx)[0].tolist())
ancIdx.extend(np.where(surf[:, 0] == maxx)[0].tolist())
ancIdx.extend(np.where(surf[:, 0] == -0.5)[0].tolist())
ancIdx.extend(np.where(surf[:, 0] == 0.5)[0].tolist())
ancIdx.extend(np.where(surf[:, 0] == -4.5)[0].tolist())
ancIdx.extend(np.where(surf[:, 0] == 4.5)[0].tolist())
# Anchors
anchors = surf_t[ancIdx, :]
print('There are %d anchors' % len(ancIdx))


# Solve
mesh = deform_mesh(mesh_src, anchors, ancIdx, 10.0)
estm_tgt = mesh.vertex

err = np.linalg.norm(estm_tgt - surf_t, axis=1)

fig = plt.figure(20)
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(estm_tgt[:,0], estm_tgt[:,1], estm_tgt[:,2], triangles=tri)
ax.plot_trisurf(surf_t[:,0], surf_t[:,1], surf_t[:,2], triangles=tri)
#ax.view_init(elev=10, azim=-90)
#plt.axis('equal')
plt.show(block=False)

fig = plt.figure(21)
ax = fig.add_subplot(111, projection='3d')
trisurf = ax.plot_trisurf(estm_tgt[:,0], estm_tgt[:,1], estm_tgt[:,2], triangles=tri)
#ax.plot_trisurf(surf_t[:,0], surf_t[:,1], surf_t[:,2], triangles=tri)
#ax.view_init(elev=10, azim=-90)
#plt.axis('equal')
plt.show(block=False)

print(err[ancIdx])

mesh_tgt.save('mesh_target.obj')
mesh_src.save('mesh_deform.obj')

"""


# --------------------------------------------------------
# Optimization
# --------------------------------------------------------
minx = min(surf[:, 0])
maxx = max(surf[:, 0])

#idx = np.where(surf[:, 0] != 0.0)[0].tolist()
idx = np.where(surf[:, 0] == minx)[0].tolist()
idx.extend(np.where(surf[:, 0] == maxx)[0].tolist())
idx.extend(np.where(surf[:, 0] == 0.5)[0].tolist())
idx.extend(np.where(surf[:, 0] == -0.5)[0].tolist())
K = len(idx)
ridx = [k for k in range(K)]
cidx = idx
data = [1.0] * K
M = sparse.coo_matrix((data, (ridx, cidx)), shape=(K, N), dtype=np.float32)











# Solve anchor
target_anchors = M.dot(surf_t)
estm_tgt, p = deform_anchor(surf, target_anchors, M, Lap, 0.0)
estm_tgt_reg, p = solve_anchor_transform(surf, target_anchors, M, Lap, 0.0000006) # cotan 0.0000006



err = estm_tgt - surf_t
err_reg = estm_tgt_reg - surf_t

#print('Error at anchor')
#print(M @ err_reg)

# Draw - Results

fig = plt.figure(10)
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(estm_tgt[:,0], estm_tgt[:,1], estm_tgt[:,2], triangles=tri)
ax.plot_trisurf(surf_t[:,0], surf_t[:,1], surf_t[:,2], triangles=tri)
#plt.axis('equal')
plt.show(block=False)


fig = plt.figure(11)
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(estm_tgt_reg[:,0], estm_tgt_reg[:,1], estm_tgt_reg[:,2], triangles=tri)
ax.plot_trisurf(surf_t[:,0], surf_t[:,1], surf_t[:,2], triangles=tri)
ax.view_init(elev=10, azim=-90)
#plt.axis('equal')
plt.show(block=True)
"""
fig = plt.figure(12)
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(estm_tgt_transform_reg[:,0], estm_tgt_transform_reg[:,1], estm_tgt_transform_reg[:,2], triangles=tri)
ax.plot_trisurf(surf_t[:,0], surf_t[:,1], surf_t[:,2], triangles=tri)
#ax.view_init(elev=10, azim=-90)
#plt.axis('equal')
plt.show(block=True)
"""

"""
color_estm = (21 / 255, 78 / 255, 108 / 255)
color_tgt = (255 / 255, 146 / 255, 41/ 255)
trimesh_estm = mlab.triangular_mesh(estm_tgt_reg[:,0], estm_tgt_reg[:,1], estm_tgt_reg[:,2], tri, color=color_estm)
trimesh_tgt = mlab.triangular_mesh(surf_t[:,0], surf_t[:,1], surf_t[:,2], tri, color=color_tgt)
mlab.view(0, 0)
mlab.show()
"""



ptrue = surf_t - surf

smoothness_true = ptrue.transpose() @ (Lap.dot(ptrue))
smoothness_estm = p.transpose() @ (Lap.dot(p))

print('Smoothness true')
print(smoothness_true)
print('Smoothness est')
print(smoothness_estm)



a = 0
