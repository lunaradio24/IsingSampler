import torch
import numpy as np
from scipy import special
from ising_model import IsingModel
from main import decimal_to_binary_tensor


class ModelPartition:
    def __init__(self, model):
        self.model = model
        self.num_spins = self.model.num_spins
        self.num_cols = self.model.num_cols
        self.num_rows = self.model.num_rows
        self.num_edges = self.model.num_edges
        self.J = self.model.J
        self.phi, self.Jdiag, self.A = self.prep_Z_anal()

    def get_Z_true(self, beta):
        """calculate the partition function of Ising model for a given beta """
        num_state = 2 ** self.num_spins
        Z = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 1 - 2 * binary
            H = self.model.get_hamiltonian(spin_state)
            Z += torch.exp(- beta * H)
        return Z


    def get_energy_true(self, beta):
        """calculate the expectation value of the Energy of Ising model with Boltzmann distribution """
        num_state = 2 ** self.num_spins
        Z = self.get_Z_true(beta)
        energy = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 1 - 2 * binary
            H = self.model.get_hamiltonian(spin_state)
            energy += H * torch.exp(- beta * H)
        return energy / Z

    def get_m_square_true(self, beta):
        """calculate the expectation value of square of the magnetisation in Ising model with Boltzmann distribution """
        num_state = 2 ** self.num_spins
        Z = self.get_Z_true(beta)
        m_square = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 1 - 2 * binary
            m = self.get_magnetisation(spin_state)
            H = self.get_hamiltonian(spin_state)
            m_square += m * m * torch.exp(- beta * H)
        return m_square / Z

    def get_entropy_true(self, beta):
        """calculate the entropy of Ising model with Boltzmann distribution """
        Z = self.get_Z_true(beta).unsqueeze(0)
        return beta * self.get_energy_true(beta) + torch.log(Z)


    def get_energy_onsager(self, beta, interaction):
        """Onsager solution; analytic solution of energy for the infinite 2d square Ising lattice """
        if interaction == 'ferromagnetic':
            J = 1.
        elif interaction == 'anti-ferromagnetic':
            J = -1.
        else:
            raise Exception("invalid interaction type for this function")
        b = 2 * beta * J
        a = 2 * np.tanh(b) / np.cosh(b)
        K = special.ellipk(a ** 2)
        U = - J / np.tanh(b) * (1 + 2 / np.pi * (2 * np.tanh(b) ** 2 - 1) * K) / 2
        return U


    def prep_Z_anal(self):
        """preparation procedure for calculation of Z_analytic """
        L = self.num_cols
        nedge = self.num_edges
        J = self.J

        # angle between two (different) connected edges; set default angle = 1 which means two edges are disconnected
        phi = torch.ones(2 * nedge, 2 * nedge)

        # tensor A which is required when calculating W=AD in Z_analytic
        A = torch.zeros((2 * nedge, 2 * nedge), dtype=torch.cfloat)

        # reform (nspin, nspin) matrix J[i,j] to a diagonal (2*nedge, 2*nedge) matrix J[ij,ij]; required to get a tensor D in W=AD
        Jdiag = torch.zeros((2 * nedge, 2 * nedge), dtype=torch.cfloat)
        for e1 in range(2 * nedge):
            if e1 < nedge:
                i, j = self.model.get_vertices(e1)            # e1 = (i,j) is a directed edge from i to j
                Jdiag[e1, e1] = J[i, j]
            else:
                j, i = self.model.get_vertices(e1 - nedge)    # the inversed e1 above; i here is j above and j here is i above
                Jdiag[e1, e1] = J[i, j]
            for e2 in range(2 * nedge):
                if e2 < nedge:
                    k, l = self.model.get_vertices(e2)            # e2 = (k,l) is a directed edge from i to j
                else:
                    l, k = self.model.get_vertices(e2 - nedge)    # the inversed e2 above; k here is l above and l here is k above
                # phi[e1,e2] means the angle between two edges e1=(i,j) and e2=(k,l) when j==k and i!=l
                if j == k and i != l:
                    if (i==j-1 and l==j-L) or (i==j-L and l==j+1) or (i==j+1 and l==j+L) or (i==j+L and l==j-1):
                        phi[e1, e2] = np.pi / 2
                    elif (l==j-1 and i==j-L) or (l==j-L and i==j+1) or (l==j+1 and i==j+L) or (l==j+L and i==j-1):
                        phi[e1, e2] = -np.pi / 2
                    elif i-l== -2 or i-l==2 or i-l== -2*L or i-l==2*L:
                        phi[e1, e2] = 0
                    # get an element A[e1, e2] in terms of phi[e1, e2]
                    A[e1, e2] = np.exp(0.5 * 1j * phi[e1, e2])
        return phi, Jdiag, A


    def get_Z_anal(self, beta):
        """analytic solution of partition function for 2d square Ising lattice even with the random bond J = 1 or -1 """
        nspin = self.num_spins
        nedge = self.num_edges
        A = self.A
        theta = - beta * self.J
        D = torch.tanh(- beta * self.Jdiag)
        W = torch.matmul(A,D)
        I = torch.eye(2 * nedge)
        PIproduct = 1
        for i in range(nspin):
            for j in range(i, nspin):
                if theta[i,j] != 0:
                    PIproduct *= np.cosh(theta[i,j])
        Z = pow(2,nspin) * PIproduct * np.sqrt(torch.det(I-W))
        return Z.real


    def get_energy_anal(self, beta):
        """calculate energy from taking beta-derivative of Z_analytic """
        nspin = self.num_spins
        nedge = self.num_edges
        A = self.A
        theta = - beta * self.J
        D = torch.tanh(- beta * self.Jdiag)
        W = torch.matmul(A, D)
        Jdiag = self.Jdiag
        I = torch.eye(2 * nedge)
        term1 = 0
        for i in range(nspin):
            for j in range(i, nspin):
                if theta[i, j] != 0:
                    term1 += np.tanh(theta[i, j]) * self.J[i, j]
        tr_input1 = torch.matmul(torch.inverse(I-W), A)
        tr_input2 = torch.matmul(I-torch.tanh(-beta * Jdiag)**2, Jdiag)
        term2 = -0.5 * torch.trace(torch.matmul(tr_input1, tr_input2))
        E = term1 + term2
        return E.real
