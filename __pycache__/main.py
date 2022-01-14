import torch
import numpy as np
import pylab
from scipy import special
from typing import Tuple
import numba as nb
from tqdm import tqdm
import sys

from ising_model import IsingModel
from sampler import SampleWithRBM, sample_by_single_flip


def decimal_to_binary_tensor(value, width=0):
    string = format(value, '0{}b'.format(width))
    binary = [0 if c == '0' else 1 for c in string]
    return torch.tensor(binary, dtype=torch.float)


class AnalyzeData:
    def get_prob_data(self, dataset):
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


    def get_entropy_data(self, dataset):
        """calculate the Entropy of the dataset """
        prob = self.get_prob_data(dataset)
        data_length = dataset.size()[1]
        num_state = 2 ** data_length
        entropy = 0.
        for i_state in range(num_state):
            if prob[i_state] > 0:
                entropy -= prob[i_state] * torch.log(prob[i_state])
        return entropy


    def get_energy_data(self, model, dataset):
        """calculate the average of the Energy from the dataset """
        spinset = 1 - 2 * dataset
        E = model.get_hamiltonian(spinset)
        return np.mean(E)


    def get_magnetisation_data(self, dataset):
        """calculate the average of the magnetization from the dataset """
        spinset = 1 - 2 * dataset
        m = torch.mean(spinset, dim=1)
        return torch.mean(m)


    def get_m_square_data(self, dataset):
        """calculate the average of the m_square from the dataset """
        spinset = 1 - 2 * dataset
        m = torch.mean(spinset, dim=1)
        return torch.mean(m ** 2)


    def get_correlation(self, function, step_rough: int, effbreak):
        """calculate the estimate of autocorrelation for energy

        Args:
            function ([type]): [description]
            step_rough (int): It should be smaller than N. It is just for finding M << N s.t corr[M] converges to zero

        Returns:
            [type]: [description]
        """

        N = function.size()[0]
        mean = torch.mean(function)
        corr = torch.zeros(step_rough)
        if effbreak == True:
            step_eff = 0
        else:
            step_eff = step_rough


        for t in range(step_rough):
            data_0 = function[0:N-t] - mean
            data_t = function[t:] - mean
            corr[t] += np.dot(data_0, data_t) / (N-t)

            #for n in range(N - t):
            #    corr[t] += (function[n] - mean) * (function[n + t] - mean) / (N - t)
            
            if effbreak == True and corr[t]/corr[0] < 0.01:     # stop and break the loop when correlation becomes small enough
                step_eff = t+1                                  # return t as the effective number of steps M << N
                break
        
        return corr, step_eff


    def get_autocorrtime(self, corr, step_eff):
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
        self.model = IsingModel(num_rows, num_cols, interaction)
        self.rbm_sampler = SampleWithRBM(self.model, ensemble_size, num_rows, num_cols, interaction)
        self.anal = AnalyzeData()


    def get_energy_in_terms_of_num_steps(self, beta, step_max, sampling_method, sample_initial):
        energy = torch.zeros(step_max)
        sample = sample_initial
        # sampling by Ising-RBM mapping
        if sampling_method == 'rbm-mapping':
            for step_i in range(step_max):
                energy[step_i] = self.anal.get_energy_data(self.model, sample.to(torch.float64).numpy())
                sample = self.rbm_sampler.sample_by_rbm_mapping(beta, sample)
        # sampling by single-flip MC
        elif sampling_method == 'single-flip':
            sample_np = sample.to(torch.float64).numpy()
            for step_i in range(step_max):
                energy[step_i] = self.anal.get_energy_data(self.model, sample_np)
                sample_np = sample_by_single_flip(self.model, self.ensemble_size, beta, sample_np)
        else:
            raise Exception("invalid sampling_method")
        return energy


    def get_energy_in_terms_of_beta(self, beta_range, step_max, sampling_method, sample_initial):
        energy = torch.zeros(len(beta_range))
        sample = sample_initial
        # sampling by Ising-RBM mapping
        if sampling_method == 'rbm-mapping':
            for i, beta in enumerate(beta_range):
                for _ in range(step_max):
                    sample = rbm_sampler.sample_by_rbm_mapping(beta, sample)
                energy[i] = self.anal.get_energy_data(self.model, sample.to(torch.float64).numpy())
        # sampling by single-flip MC
        elif sampling_method == 'single-flip':
            for i, beta in enumerate(beta_range):
                sample_np = sample.to(torch.float64).numpy()
                for _ in range(step_max):
                    sample_np = sample_by_single_flip(self.model, self.ensemble_size, beta, sample_np)
                energy[i] = self.anal.get_energy_data(self.model, sample_np)
        else:
            raise Exception("invalid sampling_method")
        return energy


    def plot_energy_over_beta(self, beta_range, step_max, interaction, sampling_methods, sample_initial):
        """plot energy_sample(after converged) and energy_true(or energy_anal if it's correct) in terms of beta """
        pylab.figure(1)
        E_anal = np.zeros(len(beta_range))
        #E_true = np.zeros(len(beta_range))
        #E_onsager = np.zeros(len(beta_range))

        for i, beta in enumerate(beta_range):
            E_anal[i] = self.model.get_energy_anal(beta)
            #E_true[i] = self.model.get_energy_true(beta)

        E_sample1 = self.get_energy_in_terms_of_beta(beta_range, step_max[0], sampling_methods[0], sample_initial).numpy()
        E_sample2 = self.get_energy_in_terms_of_beta(beta_range, step_max[1], sampling_methods[1], sample_initial).numpy()
        #E_onsager = self.model.get_energy_onsager(beta_range, interaction)

        #pylab.plot(beta_range, E_true/self.num_edges, label='E_true')
        #pylab.plot(beta_range, E_onsager, label='E_onsager')
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


    def plot_corr_over_mcstep(self, step_max, step_rough, sampling_method, sample_initial):
        """plot autocorrelation for energy in terms of the number of Markov-chain steps for a given beta """
        beta_start, beta_end, beta_step = 0.25, 1.00, 0.10
        pylab.figure(1)
        x = np.arange(step_rough)
        for beta in np.arange(beta_start, beta_end, beta_step):
            energy = self.get_energy_in_terms_of_num_steps(beta, step_max, sampling_method, sample_initial)
            corr = self.anal.get_correlation(energy / self.num_edges, step_rough, effbreak=False)[0]
            corr_normalized = corr / corr[0]
            y = corr_normalized.numpy()
            pylab.plot(x, y, label="%.2f"% beta)
        pylab.xlabel('number of Markov-chain steps')
        pylab.ylabel('autocorrelation for energy per edge')
        pylab.title('Autocorrelation in %d x %d %s Ising lattice' % (self.num_rows, self.num_cols, self.interaction))
        pylab.legend(title='beta', loc='upper right')
        pylab.show()


    def plot_tau_over_beta(self, beta_range, step_max, step_rough, sampling_method, sample_initial):
        """plot autocorrelation time for energy in terms of Markov step for a given beta """
        pylab.figure(1)
        tau = []
        for beta in beta_range:
            energy = self.get_energy_in_terms_of_num_steps(beta, step_max, sampling_method, sample_initial)
            corr, step_eff = self.anal.get_correlation(energy / self.num_edges, step_rough, effbreak=True)
            tau.append(self.anal.get_autocorrtime(corr, step_eff))
        if sampling_method=='rbm-mapping':
            pylab.plot(beta_range, tau, label='Ising-RBM mapping (ensemble=%d, N=%d, M=%d)' % (self.ensemble_size, step_max, step_rough))
        elif sampling_method=='single-flip':
            pylab.plot(beta_range, tau, label='Single-flip')
        else:
            raise Exception("invalid sampling_method")
        pylab.xlabel('beta')
        pylab.ylabel('autocorrelation time for energy per edge')
        pylab.title('Autocorrelation time in %d x %d %s Ising lattice' % (self.num_rows, self.num_cols, self.interaction))
        pylab.legend(loc='upper left')
        pylab.show()

    def plot_tau_compare(self, beta_range, step_max, step_rough, sampling_method, sample_initial):
        """plot autocorrelation time for energy from two different sampling methods in terms of Markov step for a given beta """
        pylab.figure(1)
        tau1 = []
        tau2 = []
        for beta in tqdm(beta_range):
            energy1 = self.get_energy_in_terms_of_num_steps(beta, step_max[0], sampling_method[0], sample_initial)
            corr1, step_eff1 = self.anal.get_correlation(energy1 / self.num_edges, step_rough, effbreak=True)
            tau1.append(self.anal.get_autocorrtime(corr1, step_eff1))
            energy2 = self.get_energy_in_terms_of_num_steps(beta, step_max[1], sampling_method[1], sample_initial)
            corr2, step_eff2 = self.anal.get_correlation(energy2 / self.num_edges, step_rough, effbreak=True)
            tau2.append(self.anal.get_autocorrtime(corr2, step_eff2))
        print(tau1, tau2)
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
            Z_true[i] = self.model.get_Z_true(beta)
            Z_anal[i] = self.model.get_Z_anal(beta)
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
    num_rows, num_cols = 4, 4
    interaction = ['ferromagnetic', 'anti-ferromagnetic', 'random-bond']
    # set sample size
    # ensemble_size = 100
    ensemble_size = 5
    # set the number of markov-chain steps(N=step_max, step_rough) when sampling
    step_max = [50000, 50000]
    step_rough = 1000
    # set the range of beta
    #beta_start, beta_end, beta_step = 0.01, 1.0, 0.01
    beta_start, beta_end, beta_step = 0.05, 1.0, 0.05
    beta_range = np.arange(beta_start, beta_end, beta_step)
    # set initial sample and sampling methods
    sample_initial = torch.bernoulli(torch.rand(ensemble_size, num_rows * num_cols))
    sampling_methods = ['rbm-mapping', 'single-flip']

    plot = PlotData(ensemble_size, num_rows, num_cols, interaction[2])

    if plot_number == 1:
        plot.plot_energy_over_beta(beta_range, step_max, interaction[0], sampling_methods, sample_initial)
    elif plot_number == 2:
        plot.plot_energy_over_mcstep(step_max[0], sampling_methods[0], sample_initial)
    elif plot_number == 3:
        plot.plot_corr_over_mcstep(step_max[0], step_rough, sampling_methods[0], sample_initial)
    elif plot_number == 4:
        plot.plot_tau_over_beta(beta_range, step_max[0], step_rough, sampling_methods[0], sample_initial)
    elif plot_number == 5:
        plot.plot_tau_compare(beta_range, step_max, step_rough, sampling_methods, sample_initial)
    elif plot_number == 6:
        plot.plot_Z_over_beta(beta_range)
    else:
        raise Exception("invalid plot_number")

#################################################################################################################

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    main(5)
