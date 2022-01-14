from typing import Tuple, Union
import numpy as np
import numba as nb
from numba.experimental import jitclass


ising_model_spec = [
    ('num_rows', nb.int32),
    ('num_cols', nb.int32),
    ('num_spins', nb.int32),
    ('num_edges', nb.int32),
    ('J', nb.int32[:,:])
]

@jitclass(ising_model_spec)
class IsingModel:
    def __init__(self, num_rows, num_cols, interaction):
        self.num_rows, self.num_cols = num_rows, num_cols
        self.num_spins = num_rows * num_cols
        self.num_edges = 2 * num_rows * num_cols - num_rows - num_cols
        self.J = self.get_interactions(interaction)

    def get_edge(self, i: int, j: int) -> int:
        """get the label of a edge connecting given two adjacent vertices (Warning: i should be at the left side or upside of j)"""
        J = self.J
        num_horizontal_edges = (self.num_cols - 1) * self.num_rows
        e = 0
        if J[i, j] != 0:
            if i - j == self.num_cols:
                e = num_horizontal_edges + j
            elif i - j == - self.num_cols:
                e = num_horizontal_edges + i
            elif i - j == 1:
                e = (self.num_cols - 1) * int(i / self.num_cols) + i % self.num_cols - 1
            elif i - j == -1:
                e = (self.num_cols - 1) * int(i / self.num_cols) + i % self.num_cols
            return e
        elif J[i, j] == 0:
            raise Exception("site i and j are not on the same edge")
        else:
            raise Exception("invalid i and j for given num_rows and num_cols")


    def get_vertices(self, e: int) -> Tuple[int, int]:
        """get labels of two adjacent vertices corresponding to a given edge """
        num_horizontal_edges = (self.num_cols - 1) * self.num_rows
        if e < num_horizontal_edges:
            row, col = divmod(e, self.num_cols - 1)
            i = row * self.num_cols + col
            j = i + 1
            return i, j
        elif e >= num_horizontal_edges:
            i = e - num_horizontal_edges
            j = i + self.num_cols
            return i, j
        else:
            raise Exception("invalid label of edge for given num_rows and num_cols")


    def get_neighbors(self, i: int, j: int):
        """find nearest neighbors for a given vertex at site (i,j) in 2d lattice """
        nhb = []
        if i > 0:
            nhb.append([i - 1, j])
        if i < self.num_rows - 1:
            nhb.append([i + 1, j])
        if j > 0:
            nhb.append([i, j - 1])
        if j < self.num_cols - 1:
            nhb.append([i, j + 1])
        return np.array(nhb)

    def get_interactions(self, interaction):
        """get a matrix of interactions only among nearest neighbors """
        nspin = self.num_spins
        L = self.num_cols
        J = np.zeros((nspin, nspin), dtype = np.int32)
        for s1 in range(nspin):
            i, j = divmod(s1, L)
            for i_nhb, j_nhb in self.get_neighbors(i, j):
                s2 = L * i_nhb + j_nhb
                if J[s1, s2] == 0:
                    if interaction == 'ferromagnetic':
                        J[s1, s2] = 1
                    elif interaction == 'anti-ferromagnetic':
                        J[s1, s2] = -1
                    elif interaction == 'random-bond':
                        J[s1, s2] = 1 - 2 * np.random.choice(2)
                    else:
                        raise Exception("invalid interaction type")
                    J[s2, s1] = J[s1, s2]
        return J


    def get_hamiltonian(self, spin_state):
        """calculate Hamiltonian of Ising model for a given spin state """

        J = self.J.astype(np.float64)

        spin_state = spin_state.reshape(-1, self.num_spins) # to 2d array

        H = []
        for s in spin_state:
            H.append(- np.dot(s, np.dot(J, s)) / 2)
        return np.array(H)

    def get_magnetisation(self, spin_state):
        """calculate magnetisation of Ising model for a given spin state """
        m = 0.
        if spin_state.dim() == 1:
            m = np.mean(spin_state)
        elif spin_state.dim() == 2:
            m = np.mean(spin_state, dim=1)
        return m


