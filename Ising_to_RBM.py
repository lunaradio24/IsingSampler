import torch
import numpy as np
import pylab
from scipy import special
from typing import Tuple


def decimal_to_binary_tensor(value, width=0):
    string = format(value, '0{}b'.format(width))
    binary = [0 if c == '0' else 1 for c in string]
    return torch.tensor(binary, dtype=torch.float)


class IsingModel:

    def __init__(self, num_rows, num_cols, interaction):
        self.num_rows, self.num_cols = num_rows, num_cols
        self.num_spins = num_rows * num_cols
        self.num_edges = 2 * num_rows * num_cols - num_rows - num_cols
        self.J = self.get_interactions(interaction)
        self.phi, self.Jdiag, self.A = self.prep_Z_anal()


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
        return nhb


    def get_interactions(self, interaction):
        """get a matrix of interactions only among nearest neighbors """
        nspin = self.num_spins
        L = self.num_cols
        J = torch.zeros(nspin, nspin)
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
                        J[s1, s2] = np.random.choice([1, -1])
                    else:
                        raise Exception("invalid interaction type")
                    J[s2, s1] = J[s1, s2]
        return J


    def hamiltonian(self, spin_state):
        """calculate Hamiltonian of Ising model for a given spin state """
        if spin_state.dim() == 1:
            s = spin_state.float().unsqueeze(0)
        else:
            s = spin_state.float()
        J = self.J
        H = - torch.matmul(s, torch.matmul(J, s.t())) / 2
        if H.dim() == 0:
            return H
        elif H.dim() == 2:
            return torch.diag(H)

    def magnetisation(self, spin_state):
        """calculate magnetisation of Ising model for a given spin state """
        m = 0.
        if spin_state.dim() == 1:
            m = torch.mean(spin_state)
        elif spin_state.dim() == 2:
            m = torch.mean(spin_state, dim=1)
        return m

    def partition_function(self, beta):
        """calculate the partition function of Ising model for a given beta """
        num_state = 2 ** self.num_spins
        Z = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 1 - 2 * binary
            H = self.hamiltonian(spin_state)
            Z += torch.exp(- beta * H)
        return Z


    def energy_true(self, beta):
        """calculate the expectation value of the Energy of Ising model with Boltzmann distribution """
        num_state = 2 ** self.num_spins
        Z = self.partition_function(beta)
        energy = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 1 - 2 * binary
            H = self.hamiltonian(spin_state)
            energy += H * torch.exp(- beta * H)
        return energy / Z

    def m_square_true(self, beta):
        """calculate the expectation value of square of the magnetisation in Ising model with Boltzmann distribution """
        num_state = 2 ** self.num_spins
        Z = self.partition_function(beta)
        m_square = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 1 - 2 * binary
            m = self.magnetisation(spin_state)
            H = self.hamiltonian(spin_state)
            m_square += m * m * torch.exp(- beta * H)
        return m_square / Z

    def entropy_true(self, beta):
        """calculate the entropy of Ising model with Boltzmann distribution """
        Z = self.partition_function(beta).unsqueeze(0)
        return beta * self.energy_true(beta) + torch.log(Z)


    def onsagar_solution(self, beta, interaction):
        """Onsagar solution; analytic solution of energy for the infinite 2d square Ising lattice """
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
                i, j = self.get_vertices(e1)            # e1 = (i,j) is a directed edge from i to j
                Jdiag[e1, e1] = J[i, j]
            else:
                j, i = self.get_vertices(e1 - nedge)    # the inversed e1 above; i here is j above and j here is i above
                Jdiag[e1, e1] = J[i, j]
            for e2 in range(2 * nedge):
                if e2 < nedge:
                    k, l = self.get_vertices(e2)            # e2 = (k,l) is a directed edge from i to j
                else:
                    l, k = self.get_vertices(e2 - nedge)    # the inversed e2 above; k here is l above and l here is k above
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


    def Z_analytic(self, beta):
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


    def energy_analytic(self, beta):
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



class RBM:


    def __init__(self, num_rows: int, num_cols: int, interaction):
        self.num_rows, self.num_cols = num_rows, num_cols
        self.num_v = num_rows * num_cols
        self.num_h = 2 * num_rows * num_cols - num_rows - num_cols
        self.model = IsingModel(num_rows, num_cols, interaction)
        self.W = torch.zeros(self.num_v, self.num_h)
        self.a = torch.zeros(self.num_v)
        self.b = torch.zeros(self.num_h)


    def get_weights(self, beta):
        """calculate RBM parameters from a given visible layer by Ising-RBM mapping """
        J = self.model.J
        self.W = torch.zeros(self.num_v, self.num_h)
        for e in range(self.num_h):
            (i, j) = self.model.get_vertices(e)
            self.W[i, e] = 2 * np.arccosh(np.exp(2 * beta * abs(J[i, j])))
            self.W[j, e] = 2 * np.arccosh(np.exp(2 * beta * abs(J[i, j]))) * np.sign(J[i, j])

    def get_biases_a(self):
        """obtain the bias_a from the weight w by summing all rows along columns """
        self.a = - 0.5 * torch.sum(self.W, dim=1)

    def get_biases_b(self):
        """obtain the bias_b from the weight w by summing all columns along rows """
        self.b = - 0.5 * torch.sum(self.W, dim=0)

    def to_hidden(self, visible):
        """get a new hidden layer from a given visible layer using sigmoid function """
        b = self.b
        vW = torch.matmul(visible, self.W)
        return torch.sigmoid(vW + b)

    def to_visible(self, hidden):
        """get a new visible layer from a given hidden layer using sigmoid function """
        a = self.a
        Wh = torch.matmul(hidden, self.W.t())
        return torch.sigmoid(Wh + a)


class IsingSampler:

    def __init__(self, ensemble_size: int, num_rows: int, num_cols: int, interaction):
        self.ensemble_size = ensemble_size
        self.num_rows, self.num_cols = num_rows, num_cols
        self.rbm = RBM(num_rows, num_cols, interaction)
        self.model = self.rbm.model


    def gen_random_sample(self, p):
        """randomly generate one sample of an spin state as 2d tensor """
        probs = torch.rand(self.ensemble_size, self.num_rows * self.num_cols)
        sample = torch.where(probs < p, torch.zeros(1), torch.ones(1))
        return sample


    def get_RBM_parameters(self, beta):
        """calculate RBM-parameters by using a mapping from Ising model to RBM """
        self.rbm.get_weights(beta)
        self.rbm.get_biases_a()
        self.rbm.get_biases_b()


    def rbm_mapping(self, sample_initial):
        """gibbs sampling with conditional probs from Ising-RBM mapping: one step """
        v_current = sample_initial
        pr_h = self.rbm.to_hidden(v_current)
        h_next = torch.bernoulli(pr_h)
        pr_v = self.rbm.to_visible(h_next)
        v_next = torch.bernoulli(pr_v)
        return v_next


    def single_flip(self, beta, sample_initial):
        """sampling by metropolis Monte Carlo method with single-flip """
        spins = 1 - 2 * sample_initial
        # pick a random spin
        i = np.random.randint(0, self.num_rows - 1)
        j = np.random.randint(0, self.num_cols - 1)
        n = self.num_cols * i + j
        # calculate the energy difference between before and after flip
        dE = torch.zeros(self.ensemble_size)
        for i_nhb, j_nhb in self.model.get_neighbors(i, j):
            n_nhb = self.num_cols * i_nhb + j_nhb
            J = self.model.J[n, n_nhb]
            dE += 2 * J * spins[:, n] * spins[:, n_nhb]
        # generate an uniform random number
        r = torch.rand(self.ensemble_size)
        # flip the spin if r < acceptance probability min(1, exp(-beta*dE))
        spins[:, n] = torch.where(r < torch.exp(-beta * dE), -spins[:, n], spins[:, n])
        sample_final = (1-spins)/2
        return sample_final


class AnalyzeData:

    def prob(self, dataset):
        """calculate the probability of each samples in the dataset by counting """
        data_size = dataset.size()[0]
        data_length = dataset.size()[1]
        dataset = dataset
        num_state = 2 ** data_length
        count = torch.zeros(num_state)
        for i_state, j_data in iter.product(range(num_state), range(data_size)):
            bin_state = decimal_to_binary_tensor(i_state, width=data_length)
            if torch.all(torch.eq(dataset[j_data], bin_state)) == 1:
                count[i_state] += 1
        prob = count / data_size
        return prob


    def entropy(self, dataset):
        """calculate the Entropy of the dataset """
        prob = self.prob(dataset)
        data_length = dataset.size()[1]
        num_state = 2 ** data_length
        entropy = 0.
        for i_state in range(num_state):
            if prob[i_state] > 0:
                entropy -= prob[i_state] * torch.log(prob[i_state])
        return entropy


    def energy(self, model, dataset):
        """calculate the average of the Energy from the dataset """
        spinset = 1 - 2 * dataset
        E = model.hamiltonian(spinset)
        return torch.mean(E)


    def magnetisation(self, dataset):
        """calculate the average of the magnetization from the dataset """
        spinset = 1 - 2 * dataset
        m = torch.mean(spinset, dim=1)
        return torch.mean(m)


    def m_square(self, dataset):
        """calculate the average of the m_square from the dataset """
        spinset = 1 - 2 * dataset
        m = torch.mean(spinset, dim=1)
        return torch.mean(m ** 2)


    def correlation(self, function, step_eff):
        """calculate the estimate of autocorrelation for energy """
        N = function.size()[0]
        M = step_eff  # effective number of steps M << N
        corr = torch.zeros(M)
        mean = torch.mean(function)
        for t in range(M):
            for n in range(N - t):
                corr[t] += (function[n] - mean) * (function[n + t] - mean) / (N - t)
        return corr


    def autocorrtime(self, corr, step_eff):
        """calculate the estimate of autocorrelation time for energy """
        tau = 0.
        for i in range(step_eff):
            if i == 0:
                tau += corr[i]/corr[0]
            else:
                tau += 2 * corr[i]/corr[0]
        return tau


class PlotData:

    def __init__(self, ensemble_size, num_rows, num_cols, interaction):
        self.num_rows, self.num_cols = num_rows, num_cols
        self.num_edges = 2 * num_rows * num_cols - num_rows - num_cols
        self.interaction = interaction
        self.ensemble_size = ensemble_size
        self.sampler = IsingSampler(ensemble_size, num_rows, num_cols, interaction)
        self.model = self.sampler.model
        self.anal = AnalyzeData()


    def get_energy_in_terms_of_num_steps(self, beta, step_max, sampling_method, sample_initial):
        energy = torch.zeros(step_max)
        sample = sample_initial
        # sampling by Ising-RBM mapping
        if sampling_method == 'rbm-mapping':
            self.sampler.get_RBM_parameters(beta)
            for step_i in range(step_max):
                energy[step_i] = self.anal.energy(self.model, sample)
                sample = self.sampler.rbm_mapping(sample)
        # sampling by single-flip MC
        elif sampling_method == 'single-flip':
            for step_i in range(step_max):
                energy[step_i] = self.anal.energy(self.model, sample)
                sample = self.sampler.single_flip(beta, sample)
        else:
            raise Exception("invalid sampling_method")
        return energy


    def get_energy_in_terms_of_beta(self, beta_range, step_max, sampling_method, sample_initial):
        energy = torch.zeros(len(beta_range))
        sample = sample_initial
        # sampling by Ising-RBM mapping
        if sampling_method == 'rbm-mapping':
            for i, beta in enumerate(beta_range):
                self.sampler.get_RBM_parameters(beta)
                for _ in range(step_max):
                    sample = self.sampler.rbm_mapping(sample)
                energy[i] = self.anal.energy(self.model, sample)
        # sampling by single-flip MC
        elif sampling_method == 'single-flip':
            for i, beta in enumerate(beta_range):
                for _ in range(step_max):
                    sample = self.sampler.single_flip(beta, sample)
                energy[i] = self.anal.energy(self.model, sample)
        else:
            raise Exception("invalid sampling_method")
        return energy


    def plot_energy_over_beta(self, beta_range, step_max, interaction, sampling_methods, sample_initial):
        """plot energy_sample(after converged) and energy_true(or energy_anal if it's correct) in terms of beta """
        pylab.figure(1)
        E_anal = np.zeros(len(beta_range))
        #E_true = np.zeros(len(beta_range))
        #E_onsagar = np.zeros(len(beta_range))

        for i, beta in enumerate(beta_range): # enumerate 를 사용하여 i 정의하는거 줄여보기
            E_anal[i] = self.model.energy_analytic(beta)
            #E_true[i] = self.model.energy_true(beta)

        E_sample1 = self.get_energy_in_terms_of_beta(beta_range, step_max[0], sampling_methods[0], sample_initial).numpy()
        E_sample2 = self.get_energy_in_terms_of_beta(beta_range, step_max[1], sampling_methods[1], sample_initial).numpy()
        #E_onsagar = self.model.onsagar_solution(beta_range, interaction)

        #pylab.plot(beta_range, E_true/self.num_edges, label='E_true')
        #pylab.plot(beta_range, E_onsagar, label='E_onsagar')
        pylab.plot(beta_range, E_anal/self.num_edges, linestyle='dashed', label='E_analytic')
        pylab.plot(beta_range, E_sample1/self.num_edges, label='E_sample_rbm (ensemble=%d, MC steps=%d)' % (self.ensemble_size, step_max[0]))
        pylab.plot(beta_range, E_sample2/self.num_edges, label='E_sample_singleflip (ensemble=%d, MC steps=%d)' % (self.ensemble_size, step_max[1]))
        pylab.xlabel('beta')
        pylab.ylabel('energy per edge')
        pylab.title('Energy per edge in %d x %d %s Ising lattice' % (self.num_rows, self.num_cols, interaction))
        pylab.legend(loc='upper right')
        pylab.show()


    def plot_energy_over_mcstep(self, step_max, sampling_method, sample_initial):
        """plot energy of the sample in terms of the number of Markov-chain steps as changing beta """
        beta_start, beta_end, beta_step = 0.24, 0.74, 0.10
        pylab.figure(1)
        x = np.arange(step_max)
        for beta in np.arange(beta_start, beta_end, beta_step):
            y = self.get_energy_in_terms_of_num_steps(beta, step_max, sampling_method, sample_initial).numpy()
            pylab.plot(x, y/self.num_edges, label="%.2f"% beta)
        pylab.xlabel('number of Markov-chain steps')
        pylab.ylabel('energy per edge')
        pylab.title('Energy per edge in %d x %d %s Ising lattice' % (self.num_rows, self.num_cols, self.interaction))
        pylab.legend(title='beta', loc='upper right')
        pylab.show()


    def plot_corr_over_mcstep(self, step_max, step_eff, sampling_method, sample_initial):
        """plot autocorrelation for energy in terms of the number of Markov-chain steps for a given beta """
        beta_start, beta_end, beta_step = 0.25, 1.00, 0.10
        pylab.figure(1)
        x = np.arange(step_eff)
        for beta in np.arange(beta_start, beta_end, beta_step):
            energy = self.get_energy_in_terms_of_num_steps(beta, step_max, sampling_method, sample_initial)
            corr = self.anal.correlation(energy / self.num_edges, step_eff)
            corr_normalized = corr / corr[0]
            y = corr_normalized.numpy()
            pylab.plot(x, y, label="%.2f"% beta)
        pylab.xlabel('number of Markov-chain steps')
        pylab.ylabel('autocorrelation for energy per edge')
        pylab.title('Autocorrelation in %d x %d %s Ising lattice' % (self.num_rows, self.num_cols, self.interaction))
        pylab.legend(title='beta', loc='upper right')
        pylab.show()


    def plot_tau_over_beta(self, beta_range, step_max, step_eff, sampling_method, sample_initial):
        """plot autocorrelation time for energy in terms of Markov step for a given beta """
        pylab.figure(1)
        tau = []
        for beta in beta_range:
            energy = self.get_energy_in_terms_of_num_steps(beta, step_max, sampling_method, sample_initial)
            corr = self.anal.correlation(energy / self.num_edges, step_eff)
            tau.append(self.anal.autocorrtime(corr, step_eff))
        if sampling_method=='rbm-mapping':
            pylab.plot(beta_range, tau, label='Ising-RBM mapping (ensemble=%d, N=%d, M=%d)' % (self.ensemble_size, step_max, step_eff))
        elif sampling_method=='single-flip':
            pylab.plot(beta_range, tau, label='Single-flip')
        else:
            raise Exception("invalid sampling_method")
        pylab.xlabel('beta')
        pylab.ylabel('autocorrelation time for energy per edge')
        pylab.title('Autocorrelation time in %d x %d %s Ising lattice' % (self.num_rows, self.num_cols, self.interaction))
        pylab.legend(loc='upper left')
        pylab.show()


    def plot_tau_compare(self, beta_range, step_max, step_eff, sampling_method, sample_initial):
        """plot autocorrelation time for energy from two different sampling methods in terms of Markov step for a given beta """
        pylab.figure(1)
        tau1 = []
        tau2 = []
        for beta in beta_range:
            energy1 = self.get_energy_in_terms_of_num_steps(beta, step_max[0], sampling_method[0], sample_initial)
            corr1 = self.anal.correlation(energy1 / self.num_edges, step_eff[0])
            tau1.append(self.anal.autocorrtime(corr1, step_eff[0]))
            energy2 = self.get_energy_in_terms_of_num_steps(beta, step_max[1], sampling_method[1], sample_initial)
            corr2 = self.anal.correlation(energy2 / self.num_edges, step_eff[1])
            tau2.append(self.anal.autocorrtime(corr2, step_eff[1]))
        pylab.plot(beta_range, tau1, label='Ising-RBM mapping')
        pylab.plot(beta_range, tau2, label='Single-flip')
        pylab.xlabel('beta')
        pylab.ylabel('autocorrelation time for energy per edge')
        pylab.yscale('log')
        pylab.legend()
        pylab.title('Autocorrelation time in %d x %d %s Ising lattice' % (self.num_rows, self.num_cols, self.interaction))
        pylab.show()


    def plot_Z_over_beta(self, beta_range):
        """plot Z calculated from all configurations (Z_true) and from analytic solution (Z_anal) """
        pylab.figure(1)
        Z_true = torch.zeros(len(beta_range))
        Z_anal = torch.zeros(len(beta_range), dtype=torch.complex64)
        for i, beta in enumerate(beta_range):
            Z_true[i] = self.model.partition_function(beta)
            Z_anal[i] = self.model.Z_analytic(beta)
        pylab.plot(beta_range, Z_true, label='Z_true')
        pylab.plot(beta_range, Z_anal, linestyle='dashed', label='Z_anal')
        pylab.xlabel('beta')
        pylab.ylabel('Z')
        pylab.legend()
        pylab.title('partition function in %d x %d %s Ising lattice' % (self.num_rows, self.num_cols, self.interaction))
        pylab.show()

##################################################################################################################
def main(plot_number):
    # set the size and interaction type of Ising model
    num_rows, num_cols = 16, 16
    interaction = ['ferromagnetic', 'anti-ferromagnetic', 'random-bond']
    # set sample size
    ensemble_size = 1
    # set the number of markov-chain steps(N=step_max, M=step_eff) when sampling
    step_max = [10, 10]
    step_eff = [1, 1]
    # set the range of beta
    beta_start, beta_end, beta_step = 0.01, 1.0, 0.01
    beta_range = np.arange(beta_start, beta_end, beta_step)
    # set initial sample and sampling methods
    sample_initial = torch.zeros(ensemble_size, num_rows * num_cols)
    sampling_methods = ['rbm-mapping', 'single-flip']

    plot = PlotData(ensemble_size, num_rows, num_cols, interaction[0])

    if plot_number == 1:
        plot.plot_energy_over_beta(beta_range, step_max, interaction[0], sampling_methods, sample_initial)
    elif plot_number == 2:
        plot.plot_energy_over_mcstep(step_max[0], sampling_methods[0], sample_initial)
    elif plot_number == 3:
        plot.plot_corr_over_mcstep(step_max[0], step_eff[0], sampling_methods[0], sample_initial)
    elif plot_number == 4:
        plot.plot_tau_over_beta(beta_range, step_max[0], step_eff[0], sampling_methods[0], sample_initial)
    elif plot_number == 5:
        plot.plot_tau_compare(beta_range, step_max, step_eff, sampling_methods, sample_initial)
    elif plot_number == 6:
        plot.plot_Z_over_beta(beta_range)
    else:
        raise Exception("invalid plot_number")

#################################################################################################################

if __name__ == '__main__':
    main(1)
