#coding: utf-8
import utils
import numpy as np


class Mesh(object):

    def __init__(self, vertices=None, trilist=None):
        """
        Create a Mesh object from an array of vertices [Nx3] and an array of triangle [Kx3]

        :param vertices: Array of vertices
        :param trilist: Array of triangles
        """
        self.vertex = vertices
        self.tri = trilist


    def compute_laplacian(self, type):
        """
        Compute Laplacian operator for this mesh

        :param type:    Type of Laplacian (Combinatorial, Normalized, Cotan, ...)
        :return:    Degree, Adjacency matrix, Laplacian operator
        """
        if type == 'combinatorial':
            return self._combinatorial_laplacian()
        elif type == 'normalized':
            return self._normalized_laplacian()
        else:
            return None

    def _combinatorial_laplacian(self):
        """
        Compute combinatorial Laplacian, L = D - A

        :return:    Degree, Adjacency matrix, Laplacian operator
        """
        deg = self._compute_degree(self.tri)
        adj = self._compute_adjacency_matrix(self.tri)
        lap = np.diagflat(deg) - adj
        return deg, adj, lap

    def _normalized_laplacian(self):
        """
        Compute combinatorial Laplacian, L = I - D ** -0.5 @ A @ D ** -0.5

        :return:    Degree, Adjacency matrix, Laplacian operator
        """
        deg = self._compute_degree(self.tri)
        adj = self._compute_adjacency_matrix(self.tri)
        lap = np.eye(adj.shape[0], adj.shape[1], dtype=np.float32)
        lap -= np.diagflat(deg ** -0.5) @ (adj @ np.diagflat(deg ** -0.5))
        return deg, adj, lap

    def _compute_degree(self, trilist):
        """
        Compute node's degree

        :param trilist: Array of triangle
        :return:        Vertex degrees
        """
        neighbor = utils.gather_neighbour(trilist)
        return np.asarray(list(map(len, neighbor)), dtype=np.float32).reshape((len(neighbor), 1))

    def _compute_adjacency_matrix(self, trilist):
        """
        Compute adjacency matrix

        :param trilist: Array of triangles
        :return:        Adjacency matrix
        """
        neighbor = utils.gather_neighbour(trilist)
        N = len(neighbor)
        adj = np.zeros((N, N), dtype=np.float32)
        for idx, n_list in enumerate(neighbor):
            adj[idx, n_list] = 1.0
            adj[n_list, idx] = 1.0
        return adj

    @property
    def vertex(self):
        """
        Access vertex storage

        :return: Array of vertices
        """
        return self.__vertex

    @property
    def tri(self):
        """
        Access triangle storage

        :return: Array of triangle
        """
        return self.__tri

    @vertex.setter
    def vertex(self, vertices):
        """
        Set a new vertices

        :param vertices: Array of vertices to overwrite with
        """
        self.__vertex = vertices

    @tri.setter
    def tri(self, trilist):
        """
        Set a new list of triangles

        :param trilist: Array of triangles to overwrite with
        """
        self.__tri = trilist