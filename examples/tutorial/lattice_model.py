import numpy as np
from scipy import spatial
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_lattice(coords, connectivity, displacements, ax=None):
    if ax is None:
        ax = plt.gca()
    vec = coords + displacements
    lines = LineCollection(vec[connectivity], color='.7', linewidth=.7)
    ax.add_collection(lines)
    ax.set_aspect('equal')
    ax.set_ylim(-.1, 1.1)
    ax.set_xlim(-.1, np.max(vec[:, 0])+.1)
    return ax


def get_rot_tensor(cos, sin):
    R = np.array([
        [cos, sin, 0, 0],
        [-sin, cos, 0, 0],
        [0, 0, cos, sin],
        [0, 0, -sin, cos]
        ])
    return R


class TrussElement:

    def __init__(self, nodes, coords, E, A) -> None:
        self.i, self.j = nodes
        self.coords = coords  # current position vector
        self.E = E
        self.A = A

    def get_truss_matrix_local(self, L) -> np.ndarray:
        self.k = self.E*self.A/L * np.array([
            [1, 0, -1, 0],
            [0, 0, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 0]
            ])
        return self.k

    def get_matrix_global(self) -> np.ndarray:
        xi = self.coords[self.i, :]
        xj = self.coords[self.j, :]
        L = np.linalg.norm(xi-xj)
        cos, sin = (xi[0]-xj[0])/L, (xi[1]-xj[1])/L

        k_local = self.get_truss_matrix_local(L)
        R = get_rot_tensor(cos, sin)
        k_global = R.T.dot(k_local).dot(R)
        return k_global


class LatticeModel:

    def __init__(self, E, N: int = None, randomness: float = None,
                 step: float = None) -> None:
        self.E = E
        self.bcs = None
        if N is not None:
            self.create_structure(N)
        if randomness is not None:
            self.add_randomness(randomness)
        if step is not None:
            self.move_top(step)
        
    def create_structure(self, N):
        xi = np.linspace(0, 1, N)
        yi = np.linspace(0, 1, N)
        xx, yy = np.meshgrid(xi, yi)
        self.N = N
        self.coordinates = np.vstack((xx.flatten(), yy.flatten())).T
        self.nb_dofs = self.coordinates.size
        self.top_indices = np.where(self.coordinates[:, 1]==1)[0]
        self.bottom_indices = np.where(self.coordinates[:, 1]==0)[0]
        triangulation = spatial.Delaunay(self.coordinates)
        connectivity = list()
        for el in triangulation.simplices:
            connectivity.extend((edge for edge in combinations(el, 2)))
        self.connectivity = np.unique(np.array(connectivity), axis=0)

        # assign area such that total L*A = 1
        self.lengths = np.linalg.norm(self.coordinates[self.connectivity, 0]-self.coordinates[self.connectivity, 1], axis=1)
        # self.areas = np.full(self.connectivity.shape[0], 1) / np.sum(self.lengths)
        self.areas = np.full(self.connectivity.shape[0], 1/self.connectivity.shape[0])
        # create matrices
        self.U = np.zeros((self.nb_dofs, 1))
        self.U_history = np.zeros((self.nb_dofs, 1))
        self.F = np.zeros((self.nb_dofs, 1))
        self.F_history = np.zeros((self.nb_dofs, 1))
        self.K = np.zeros((self.nb_dofs, self.nb_dofs))

        return self

    def add_randomness(self, scale: float) -> None:
        self.coordinates += np.random.normal(0, scale, self.coordinates.shape) * 1/self.N
        return self

    def add_bcs(self, bcs: list) -> None:
        self.bcs = bcs
        return self

    def move_top(self, value: float) -> None:
        u_top = np.vstack((
            self.top_indices,
            np.full(self.top_indices.shape, value),
            np.full(self.top_indices.shape, 0))
                          )
        u_bottom = np.vstack((
            self.bottom_indices,
            np.full(self.bottom_indices.shape, 0),
            np.full(self.bottom_indices.shape, 0))
                             )
        self.add_bcs(np.vstack((u_top.T, u_bottom.T)))
        return self

    def end_step(self) -> None:
        self.coordinates += self.U.reshape(-1, 2)
        self.U_history += self.U
        self.U = np.zeros((self.nb_dofs, 1))
        self.F_history += self.F
        self.F = np.zeros((self.nb_dofs, 1))
        return self

    @property
    def displacements(self):
        return (self.U_history + self.U).reshape(-1, 2)

    @property
    def forces(self):
        return (self.F_history + self.F).reshape(-1, 2)
    
    def get_force_bottom(self):
        return self.forces[self.bottom_indices].sum(axis=0)

    def assemble(self) -> None:

        for el_nb, (i, j) in enumerate(self.connectivity):
            element = TrussElement((i, j), self.coordinates, self.E, self.areas[el_nb])
            k_local = element.get_matrix_global()

            self.K[2*i:2*i+2, 2*i:2*i+2] += k_local[0:2, 0:2]
            self.K[2*i:2*i+2, 2*j:2*j+2] += k_local[0:2, 2:4]
            self.K[2*j:2*j+2, 2*i:2*i+2] += k_local[2:4, 0:2]
            self.K[2*j:2*j+2, 2*j:2*j+2] += k_local[2:4, 2:4]

        self.zerorows = np.where(np.all(self.K==0, axis=1))[0]
        return self

    def apply_bcs(self) -> None:
        all_dofs = np.arange(0, self.nb_dofs)
        restrained_dofs = list()
        for (node, ux, uy) in self.bcs:
            node = int(node)
            restrained_dofs.extend([2*node, 2*node+1])
            self.U[2*node] = ux
            self.U[2*node+1] = uy

        free_dofs = np.setdiff1d(all_dofs, restrained_dofs)

        self.Kff = self.K[free_dofs, :][:, free_dofs]
        self.Kfr = self.K[free_dofs, :][:, restrained_dofs]
        self.Krf = self.K[restrained_dofs, :][:, free_dofs]
        self.Krr = self.K[restrained_dofs, :][:, restrained_dofs]

        self.Ff = self.F[free_dofs, :]
        self.Fr = self.F[restrained_dofs, :]
        self.Uf = self.U[free_dofs, :]
        self.Ur = self.U[restrained_dofs, :]

        zerorows_free = np.where(np.all(self.Kff==0, axis=1))[0]
        self.zerorows_free = zerorows_free
        self.Kff = np.delete(self.Kff, zerorows_free, axis=0)
        self.Kff = np.delete(self.Kff, zerorows_free, axis=1)

        self.free_dofs_notzero = np.delete(free_dofs, zerorows_free, axis=0)
        self.Ff = np.delete(self.Ff, zerorows_free, axis=0)

    def solve(self) -> None:
        self.K = np.zeros((self.nb_dofs, self.nb_dofs))
        self.U = np.zeros((self.nb_dofs, 1))
        self.F = np.zeros((self.nb_dofs, 1))

        self.assemble()
        self.apply_bcs()

        self.Uf = np.linalg.solve(self.Kff, self.Ff - self.Kfr.dot(self.Ur))
        self.U[self.free_dofs_notzero] = self.Uf
        self.F = self.K.dot(self.U)
        
        self.end_step()  # updates the coordinates, adds disp to history

        return self

    def plot(self, vector=None, ax: plt.Axes = None):
        if ax is None:
            ax = plt.gca()
        if vector is None:
            vector = self.coordinates

        lines = LineCollection(vector[self.connectivity], color='.7', linewidth=.7)
        ax.add_collection(lines)
        ax.set_aspect('equal')
        ax.set_ylim(-.1, 1.1)
        ax.set_xlim(-.1, np.max(vector[:, 0])+.1)
            
    def plot_deformed(self, ax=None):
        self.plot(self.coordinates + self.U.reshape(-1, 2), ax)

