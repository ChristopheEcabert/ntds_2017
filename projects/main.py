from utils import *
from mesh import *
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# How to load a mesh into the notebook
# tri = utils.load_triangulation('../data/FWTri/fw_triangulation.tri')
# mesh = utils.load_meshes('../data/FWMesh/000_shape.bs')
# neighbour = utils.gather_neighbour(tri=tri)
# n_neighbour = list(map(len, neighbour))
# plt.figure(1)
# plt.hist(n_neighbour)
# plt.show(block=False)


# Generate sample
class MeshGenerator:

    def make_plane(self, nx, ny, dx=1.0, dy=1.0):
        nvertex = nx * ny
        pts = np.zeros((nvertex, 3), dtype=np.float32)
        tri = []
        for ky in range(0, ny):
            for kx in range(0, nx):
                # Define position
                x = kx * dx
                y = ky * dy
                z = 0.0
                idx = kx + nx * ky
                pts[idx, :] = [x, y, z]
                # Define triangle
                if ky > 0 and kx > 0:
                    v2 = idx
                    v1 = idx - 1
                    v3 = idx - nx
                    v0 = v3 -1
                    tri.append([v0, v1, v3])
                    tri.append([v1, v2, v3])
        return pts, np.asarray(tri, dtype=np.int32).reshape((len(tri), 3))

    def transform(self, points, func):
        pts = np.zeros(points.shape, points.dtype)
        for i, r in enumerate(points):
            pts[i, :] = func(r)
        return pts


def modifier(point):
    x = point[0] + 5.0
    y = point[1] + 7.0
    z = 0.1 * (x ** 2.0) + 3.0
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
surf, tri = gen.make_plane(nx=5, ny=5, dx=1, dy=1)
surf_t = gen.transform(surf, modifier)
N = surf.shape[0]

# Get neighbouring vertex indexes
neighbour = gather_neighbour(tri)

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(surf[:,0], surf[:,1], surf[:,2], triangles=tri)
ax.plot_trisurf(surf_t[:,0], surf_t[:,1], surf_t[:,2], triangles=tri)
plt.show(block=False)

# --------------------------------------------------------
# Generate Laplacian
# --------------------------------------------------------

mesh_src = Mesh(surf, tri)
#Deg, Adj, Lap = mesh_src.compute_laplacian('combinatorial')
Deg, Adj, Lap = mesh_src.compute_laplacian('normalized')

# Define node's degree
#Deg = np.asarray(list(map(len, neighbour)), dtype=np.float32).reshape((len(neighbour), 1))
# Create adjacency matrix
#Adj = np.zeros((N, N), dtype=np.float32)
#for idx, n_list in enumerate(neighbour):
#    for n in n_list:
#        e0 = idx
#        e1 = n
#        Adj[e0, e1] = 1.0
#        Adj[e1, e0] = 1.0

# Sanity check
assert np.count_nonzero(Adj - Adj.transpose()) == 0
assert np.count_nonzero(Lap - Lap.T) == 0

fig, ax = plt.subplots(1,2)
ax[0].spy(Adj)
ax[1].spy(Lap)

# --------------------------------------------------------
# Optimization
# --------------------------------------------------------
M = np.zeros((N, 1), dtype=np.float32)
idx = surf[:, 0] == 0.0
M[idx] = 1.0
idx = surf[:, 0] == 4.0
M[idx] = 1.0
M = np.diagflat(M)


def solve_anchor(source, target, sel, lap, alpha):
    """
    Solve the system argmin | M ( src + t - tgt) |_F using Gauss-Newton solver

    :param source:      Surface to deform
    :param target:      Surface to reach
    :param sel:         Selected anchor
    :param lap:         Laplacian operator
    :param alpha:       Regularizer
    :return:            Estimated surface and parameters
    """
    # Parameters
    src = source
    tgt = target
    p = np.zeros(src.shape, dtype=np.float32)
    process = True
    it = 0
    max_it = 5

    # Gauss-Newton
    while process:
        # Compute error f(x)
        fx = sel @ (src + p - tgt)
        # Compute Jacobian + Linear system
        Jf = sel
        A = Jf.T @ Jf
        b = - Jf.T @ fx
        # Use lstsq since A is singular
        dp, _, _, _ = np.linalg.lstsq(A, b)
        # Update
        p += dp
        # Converged ?
        it += 1
        process = True if it < max_it else False

    # Reconstruct target
    estm_tgt = (src + p)
    return estm_tgt, p


def solve_anchor_reg(source, target, sel, lap, alpha):
    """
    Solve the system argmin | M ( src + t - tgt) | + t'Lt using Gauss-Newton solver

    :param source:      Surface to deform
    :param target:      Surface to reach
    :param sel:         Selected anchor
    :param lap:         Laplacian operator
    :param alpha:       Regularizer
    :return:            Estimated surface and parameters
    """
    # Parameters
    src = source
    tgt = target
    p = np.zeros(src.shape, dtype=np.float32)
    process = True
    it = 0
    max_it = 3
    # Gauss-Newton
    while process:
        # Compute error f(x)
        F = (src + p - tgt)
        Fx = sel @ F
        # Compute Jacobians
        Jf = sel
        A = (Jf.T @ Jf) + alpha * lap
        b = - ((Jf.T @ Fx) + alpha * (lap @ p))
        # Solve
        dp = np.linalg.solve(A, b)
        # Update
        p += dp
        # Converged ?
        it += 1
        process = True if it < max_it else False

    # Reconstruct target
    estm_tgt = (src + p).reshape((-1, 3))
    return estm_tgt, p

def solve_anchor_reg_debug(source, target, sel, lap, alpha):
    """
    Solve the system argmin | M ( src + t - tgt) | + t'Lt using Gauss-Newton solver

    :param source:      Surface to deform
    :param target:      Surface to reach
    :param sel:         Selected anchor
    :param lap:         Laplacian operator
    :param alpha:       Regularizer
    :return:            Estimated surface and parameters
    """
    # Parameters
    xs = source
    xt = target
    p = np.zeros(xs.shape, dtype=np.float32)
    # Solve

    # Compute error f(x)
    ef = (xs - xt)
    MtM = sel.T @ sel
    # Compute Linear system

    A = MtM + alpha * lap
    b = - (MtM @ ef)
    # Solve
    p = np.linalg.solve(A, b)

    # Reconstruct target
    estm_tgt = (xs + p)
    return estm_tgt, p


# Solve anchor
estm_tgt, p = solve_anchor(surf, surf_t, M, Lap, 0.0)
estm_tgt_reg, p = solve_anchor_reg_debug(surf, surf_t, M, Lap, 1.0)


p_true = surf_t - surf
print('GT smoothness (p_true)')
print(p_true.T @ (Lap @ p_true))
print('Estm smoothness (p)')
print(p.T @ (Lap @ p))

err = estm_tgt - surf_t
err_reg = estm_tgt_reg - surf_t

print('Error at anchor')
print(M @ err_reg)

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
#ax.view_init(elev=0, azim=-90)
#plt.axis('equal')
plt.show(block=True)

"""
color_estm = (21 / 255, 78 / 255, 108 / 255)
color_tgt = (255 / 255, 146 / 255, 41/ 255)
trimesh_estm = mlab.triangular_mesh(estm_tgt_reg[:,0], estm_tgt_reg[:,1], estm_tgt_reg[:,2], tri, color=color_estm)
trimesh_tgt = mlab.triangular_mesh(surf_t[:,0], surf_t[:,1], surf_t[:,2], tri, color=color_tgt)
mlab.view(0, 0)
mlab.show()
"""






a = 0