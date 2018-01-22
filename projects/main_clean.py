from utils import *
from mesh import *
from deformation import *
import numpy as np
import os
from scipy import sparse

# data root folder
data_folder = '../data'

# Load source + target meshes
tri = utils.load_triangulation(os.path.join(data_folder, 'FWTri/fw_triangulation.tri'))
meshes = utils.load_meshes(os.path.join(data_folder, 'FWMesh/115_shape.bs'), [0, 22])
# Load anchors
anchors = utils.load_anchor_point(os.path.join(data_folder, 'anchors.cfg'))

# Create meshes
mesh_xs = Mesh(meshes[0], tri)
mesh_xt = Mesh(meshes[1], tri)
N = meshes[0].shape[0]

"""
# -------------------------------------------------------------
# Deformation fields
# -------------------------------------------------------------

# Find deformation fields
# 1) create selection mask
K = len(anchors)
ridx = [k for k in range(K)]
cidx = anchors
data = [1.0] * K
M = sparse.coo_matrix((data, (ridx, cidx)), shape=(K, N), dtype=np.float32)
# 2) Comptue laplacian
_,_,Lap = mesh_xs.compute_laplacian('cotan')
# 3) Compute target
Xs = mesh_xs.vertex
Xt = M.dot(mesh_xt.vertex)
# 4) Estimate deformation field without regularization
estm_xt, d = deform_anchor(Xs, Xt, M, Lap, 0.0001)

# Error
err = np.linalg.norm(estm_xt - mesh_xt.vertex, axis=1)
e = np.mean(err)

print('Point wise error')
print(err)
print('Mean error')
print(e)

# 4) Estimate deformation field without regularization
estm_xt, d = deform_regularized_anchor(Xs, Xt, M, Lap, 0.0001)
# Error
err = np.linalg.norm(estm_xt - mesh_xt.vertex, axis=1)
e = np.mean(err)

print('Point wise error')
print(err)
print('Mean error')
print(e)

"""

# -------------------------------------------------------------
# Least-Square Mesh
# -------------------------------------------------------------
# Define only anchors around the lips
anchorsIdx = [10308, 3237, 3205, 6081, 6091,6124,8832,
              6611, 1827, 3582, 402, 3584] # Mouth region
# Inner part
# 3271, 10313, 8808, 8825, 6159, 8842, 8809, 10326
anchors = mesh_xt.vertex[anchorsIdx, :]
# Deform
mesh = deform_mesh(mesh_xs, anchors, anchorsIdx, 10.0)
estm_xt = mesh.vertex

# Error
err = np.linalg.norm(estm_xt - mesh_xt.vertex, axis=1)
e = np.mean(err)

print('Point wise error')
print(err)
print('Mean error: %f' % e)






