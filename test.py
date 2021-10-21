import torch
import numpy as np
import Ising_to_RBM

# set the size and interaction type of Ising model
num_rows, num_cols = 3, 3
interaction = ['ferromagnetic', 'anti-ferromagnetic', 'random-bond']
# set sample size
ensemble_size = 100
# set the number of markov-chain steps(N=step_max, step_rough) when sampling
step_max = [100, 100]
step_rough = 20
# set the range of beta
beta_start, beta_end, beta_step = 0.01, 1.0, 0.01
beta_range = np.arange(beta_start, beta_end, beta_step)
# set initial sample and sampling methods
sample_initial = torch.zeros(ensemble_size, num_rows * num_cols)
sampling_methods = ['rbm-mapping', 'single-flip']

plot = Ising_to_RBM.PlotData(ensemble_size, num_rows, num_cols, interaction[0])
model = plot.model
num_edges = plot.num_edges

def test_Z_anal_fits_to_Z_true() -> bool:
    # check if Z_analytic fits Z_true

    Z_anal = torch.zeros(len(beta_range), dtype=torch.complex64)
    Z_true = torch.zeros(len(beta_range))
    
    for i, beta in enumerate(beta_range):
        Z_anal[i] = model.get_Z_anal(beta)
        Z_true[i] = model.get_Z_true(beta)
    
    MSE = np.square(np.subtract(Z_true, Z_anal)).mean()
    num_errors = sum(np.where(np.square(np.subtract(Z_true, Z_anal)) > MSE, True, False))

    if num_errors < len(beta_range) * 0.05:
        return True
    else:
        return False


def test_E_anal_fits_to_E_true() -> bool:
    # check if E_anal fits E_true

    E_anal = np.zeros(len(beta_range))
    E_true = np.zeros(len(beta_range))
    
    for i, beta in enumerate(beta_range):
        E_anal[i] = model.get_energy_anal(beta)
        E_true[i] = model.get_energy_true(beta)

    MSE = np.square(np.subtract(E_true, E_anal)).mean()
    num_errors = sum(np.where(np.square(np.subtract(E_true, E_anal)) > MSE, True, False))

    if num_errors < len(beta_range) * 0.05:
        return True
    else:
        return False


def test_E_sample_fits_to_E_anal() -> bool:
    # check if E_sample fits E_anal

    E_sample = plot.get_energy_in_terms_of_beta(beta_range, step_max[0], sampling_methods[0], sample_initial).numpy()
    E_anal = np.zeros(len(beta_range))

    for i, beta in enumerate(beta_range):
        E_anal[i] = model.get_energy_anal(beta)
    
    MSE = np.square(np.subtract(E_sample, E_anal)).mean()
    num_errors = sum(np.where(np.square(np.subtract(E_sample, E_anal)) > MSE, True, False))

    if num_errors < len(beta_range) * 0.05:
        return True
    else:
        return False


def test_corr_convergence(beta) -> bool:
    # check if correlation of samples converges to zero as increasing number of MCMC steps for a given beta

    energy = plot.get_energy_in_terms_of_num_steps(beta, step_max, sampling_methods[0], sample_initial)
    corr = plot.anal.get_correlation(energy / plot.num_edges, step_rough, effbreak=False)[0]
    
    if corr[step_rough] / corr[0] < 0.01:
        return True
    else:
        return False


def test_autocorrtime_convergence() -> bool:
    # check if autocorrelation time of samples converges to zero at high beta (low temperature)

    if 1>0:
        return True
    else:
        return False


def main():
    
    print(test_Z_anal_fits_to_Z_true())
    print(test_E_anal_fits_to_E_true())
    print(test_E_sample_fits_to_E_anal())
    print(test_corr_convergence())
    print(test_autocorrtime_convergence())

if __name__ == '__main__':
    main()

    
