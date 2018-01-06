import re
import shutil
import struct
import numpy as np


def prepare_data(src, dest):
    """
    Extract FaceWarehouse meshes from dataset folder (src) and copy them into
     dest folder
    :param src:     Location of FaceWarehouse meshes
    :param dest:    Location where to export them
    """
    meshes = []
    srch = re.compile(r"_([0-9]*)/")
    # Scan folders
    for root, dirs, files in os.walk(src):
        for f in files:
            if f.endswith('bs'):
                # Extract subject ID
                fname = os.path.join(root, f)
                match = srch.search(fname)
                if match:
                    sid = int(match.groups()[0]) - 1
                    meshes.append((sid, fname[len(src):]))
                else:
                    print('Error, can not determine subject ID')
    # Copy files
    if not os.path.exists(dest):
        os.makedirs(dest)
    for m in meshes:
        sid = m[0]
        path = m[1]
        fname = path.split('/')[-1]
        fdest = dest + '%03d_' % sid + fname
        if not os.path.isfile(fdest):
            print('Copy mesh: %s' % path)
            shutil.copy2(src + path, fdest)
    print('Done')


def load_triangulation(filename):
    """
    Load triangulation file
    :param filename:    Path to triangulation file (*.tri)
    :return:            Numpy array [N x 3] holding vertices index forming
                        each triangle
    """
    tris = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            tris.append(int(parts[1]) - 1)
            tris.append(int(parts[2]) - 1)
            tris.append(int(parts[3]) - 1)
    if len(tris) != 0:
        return np.asarray(tris, dtype=np.int32).reshape(-1, 3)
    else:
        print('Error, no triangle loaded')


def load_meshes(filename, exps=None):
    """
    Load meshes from binary files
    :param filename:    Path to binary mesh file
    :param exps:        List of expression to load (indices, or None)
    :return:            List of of numpy array holding meshes K * [N x 3]
    """
    meshes = []
    with open(filename, 'rb') as f:
        # Read header
        nexp = struct.unpack('@i', f.read(4))[0]
        nvert = struct.unpack('@i', f.read(4))[0]
        nvert *= 3 # x,y,z components
        npoly = struct.unpack('@i', f.read(4))[0]
        # Select expression
        idx = range(0, nexp) if exps is None else sorted(list(set(exps)))
        # Iterate over selection
        for i in idx:
            # Move to proper section
            offset = i * nvert * 4 + 12
            f.seek(offset, 0)
            # Read vertices
            vertex = struct.unpack('@{}f'.format(nvert), f.read(nvert * 4))
            meshes.append(np.asarray(vertex, dtype=np.float32).reshape(-1, 3))
    if len(meshes) == 0:
        print('Error, can not load meshes from file: %s' % filename)
    return meshes


def save_obj(vertex, tri, filename):
    """
    Dump a mesh into an obj file
    :param vertex:      Matrix with vertex
    :param tri:         Matrix with triangle
    :param filename:    Path where to save the *.obj
    """
    with open(filename, 'w') as f:
        for v in vertex:
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for t in tri:
            f.write('f %d %d %d\n' % (t[0] + 1, t[1] + 1, t[2] + 1))


def gather_neighbour(tri):
    """
    Generate the list of neighboring vertex indexes from a triangle list
    :param tri: Triangulation matrix
    :return:    List of neighbour (list(set))
    """
    #  number of element polygon
    spoly = tri.shape[1]
    assert (spoly == 3)
    # Vertex max
    N = tri.max()
    # Define neighbor
    neighbour = [set() for _ in range(N + 1)]
    for l in tri:
        for k in range(spoly):
            v0 = l[k]
            e1 = l[(k + 1) % spoly]
            e2 = l[(k + 2) % spoly]
            neighbour[v0].add(e1)
            neighbour[v0].add(e2)
    return neighbour
