import torch
import numpy as np
import numba as nb

from ising_model import IsingModel
from rbm import RBM


def ising_to_rbm(model: IsingModel, beta: float):
    num_rows, num_cols = model.num_rows, model.num_cols
    n_visible = num_rows * num_cols
    n_hidden = 2 * num_rows * num_cols - num_rows - num_cols

    """calculate RBM parameters from a given visible layer by Ising-RBM mapping """
    W = torch.zeros(n_visible, n_hidden)
    J = model.J
    for e in range(n_hidden):
        (i, j) = model.get_vertices(e)
        W[i, e] = 2 * np.arccosh(np.exp(2 * beta * abs(J[i, j])))
        W[j, e] = 2 * np.arccosh(np.exp(2 * beta * abs(J[i, j]))) * np.sign(J[i, j])

    # Obtain the biases
    a = - 0.5 * torch.sum(W, dim=1)
    b = - 0.5 * torch.sum(W, dim=0)
    return RBM.from_weights(W, a, b)


class SampleWithRBM:
    def __init__(self, model: IsingModel, ensemble_size: int, num_rows: int, num_cols: int, interaction):
        self.ensemble_size = ensemble_size
        self.num_rows, self.num_cols = num_rows, num_cols
        self.model = model

    def gen_random_sample(self, p):
        """randomly generate one sample of an spin state as 2d tensor """
        probs = torch.rand(self.ensemble_size, self.num_rows * self.num_cols)
        sample = torch.where(probs < p, torch.zeros(1), torch.ones(1))
        return sample


    def sample_by_rbm_mapping(self, beta: float, sample_initial):
        """gibbs sampling with conditional probs from Ising-RBM mapping: one step """
        rbm = ising_to_rbm(self.model, beta)

        v_current = sample_initial
        pr_h = rbm.to_hidden(v_current)
        h_next = torch.bernoulli(pr_h)
        pr_v = rbm.to_visible(h_next)
        v_next = torch.bernoulli(pr_v)
        return v_next


@nb.jit(nopython = True)
def sample_by_single_flip(model, ensemble_size, beta, sample_initial):
    """sampling by metropolis Monte Carlo method with single-flip """
    spins = 1 - 2 * sample_initial
    # pick a random spin

    num_rows, num_cols = model.num_rows, model.num_cols
    i = np.random.randint(0, num_rows - 1)
    j = np.random.randint(0, num_cols - 1)
    n = num_cols * i + j
    # calculate the energy difference between before and after flip
    dE = np.zeros(ensemble_size)
    neighbors = model.get_neighbors(i, j)
    for idx in range(neighbors.shape[0]):
        i_nhb, j_nhb = neighbors[idx][0], neighbors[idx][1]
        n_nhb = num_cols * i_nhb + j_nhb
        J = model.J[n, n_nhb]
        dE += 2 * J * spins[:, n] * spins[:, n_nhb]
    # generate an uniform random number
    r = np.random.rand(ensemble_size)
    # flip the spin if r < acceptance probability min(1, exp(-beta*dE))
    spins[r < np.exp(-beta * dE), n] *= -1
    sample_final = (1-spins)/2
    return sample_final
