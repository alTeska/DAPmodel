# Script for Cluster data analysis of GLM - distributions and it's statistics
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# from tqdm import tqdm  #TODO
from scipy.stats import ttest_ind, wilcoxon

from DAPmodel.utils_analysis import sample_distributions, plot_distribution
from DAPmodel.utils_analysis import plot_distributions_cross, plot_mean_std
from lfimodels.hodgkinhuxley import utils

from DAPmodel import obs_params, syn_current, syn_obs_data, prior, syn_obs_stats

from delfi.utils.viz import plot_pdf
from delfi.utils.io import load_pkl

# Setting up the enviroment
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="file name")
parser.add_argument("-ii", "--ii_samples", help="nr of dist samples")
args = parser.parse_args()

if args.name is None: args.name = ''
if args.ii_samples is None: args.ii_samples = '20'

args.ii_samples = int(args.ii_samples)

# Create directory
direct_inp = 'pickle/dap_model' + args.name + '/'
direct_out = 'plots/dap_models' + args.name + '/'
if not os.path.exists(direct_out):
    print('creating directory')
    os.makedirs(direct_out)

# Load From Pickle
posteriors = load_pkl(direct_inp + 'dap_posteriors' + args.name)
prior = load_pkl(direct_inp + 'dap_prior' + args.name)
params = load_pkl(direct_inp + 'dap_params' + args.name)
labels = load_pkl(direct_inp + 'dap_labels' + args.name)

# Load the Model for simulations
M = load_pkl(direct_inp + 'dap_model' + args.name)
S = load_pkl(direct_inp + 'dap_stats' + args.name)
G = load_pkl(direct_inp + 'dap_gen' + args.name)

I = load_pkl(direct_inp + 'dap_I' + args.name)
dt = load_pkl(direct_inp + 'dap_dt' + args.name)
t_on = load_pkl(direct_inp + 'dap_t_on' + args.name)
t_off = load_pkl(direct_inp + 'dap_t_off' + args.name)


# Sampling Data From Distributions
sampl_prior, sampl_posterior = sample_distributions(prior, posteriors[-1],
                                                    M, S, ii=args.ii_samples)

# Running statistical comparison
for i in np.arange(0, len(sampl_prior)):
    print('param ' + str(i+1), ttest_ind(sampl_prior[i], sampl_posterior[i]))

for i in np.arange(0, len(sampl_prior)):
    print('param ' + str(i+1), wilcoxon(sampl_prior[i], sampl_posterior[i]))


print('Generating Plots')
# Plot distributions
g1 = plot_distribution(sampl_prior, labels)
g2 = plot_distribution(sampl_posterior, labels)
# g3, _ = plot_distributions_cross(sampl_prior, sampl_posterior)


# Results Visualization
pdf_prior, _ = plot_pdf(prior, lims=[-3, 3],figsize=(10,10), resolution=8, labels_params=labels)
pdf, _ = plot_pdf(posteriors[-1], lims=[-3, 3],figsize=(10,10), resolution=50, labels_params=labels)


# Mean and Std Visualization Prior
prior_std = prior.std
prior_mean = prior.mean
prior_means_std, _ = plot_mean_std(prior_mean, prior_std, name='prior')

# Mean and Std Visualization Posterior
posterior_std = posteriors[-1].std
posterior_mean = posteriors[-1].mean
posterior_means_std, _ = plot_mean_std(posterior_mean, posterior_std,
                                       name='posterior')

# Simulation Vizualization
x_o = syn_obs_data(I, dt, params)
x_post = syn_obs_data(I, dt, posterior_mean)
idx = np.arange(0, len(x_o['data']))

simulation, axes = plt.subplots(2, 1, figsize=(16,14))
axes[0].plot(idx, x_o['I'], c='g', label='prior')
axes[0].plot(idx, x_post['I'], label='posterior')
axes[0].legend()

axes[1].step(idx, x_o['data'], c='g', label='prior')
axes[1].step(idx, x_post['data'], label='posterior')
axes[1].legend()


# pdf_prior.savefig(direct_out + 'pdf_prior.png', labels_params=labels, bbox_inches='tight')
pdf.savefig(direct_out + 'pdf.png', labels_params=labels, bbox_inches='tight')

g1.savefig(direct_out + 'distr_by_inference_prior.png', bbox_inches='tight')
g2.savefig(direct_out + 'distr_by_inference_post.png', bbox_inches='tight')

prior_means_std.savefig(direct_out + 'prior_mean_std.png', bbox_inches='tight')
simulation.savefig(direct_out + 'simulation.png', bbox_inches='tight')
posterior_means_std.savefig(direct_out + 'posterior_means_std.png',
                            bbox_inches='tight')
