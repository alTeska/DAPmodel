import time
import argparse, os, sys

import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP, DAPSimulator
from DAPmodel import obs_params, syn_current, syn_obs_data, prior, syn_obs_stats
from DAPmodel.utils_analysis import simulate_data_distr
from lfimodels.hodgkinhuxley import utils

from delfi.inference import SNPE
from delfi.summarystats import Identity
from delfi.generator import Default
from delfi.utils.io import save, save_pkl

from lfimodels.hodgkinhuxley.HodgkinHuxleyStatsSpikes import HodgkinHuxleyStatsSpikes
from lfimodels.hodgkinhuxley.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="file name")
args = parser.parse_args()

if args.name  is None: args.name = ''
directory = 'pickle/dap_model' + args.name

if not os.path.exists(directory):
    print('creating directory')
    os.makedirs(directory)

n_rounds = 1
n_samples = 100
dt = 0.01

# picking experiments observables
observables = {'loss.lprobs', 'imputation_values', 'h1.mW', 'h1.mb', 'h2.mW',
               'h2.mb', 'weights.mW', 'weights.mb', 'means.mW0', 'means.mW1',
               'means.mb0', 'means.mb1', 'precisions.mW0', 'precisions.mW1',
               'precisions.mb0', 'precisions.mb1'}

# setting up fake parameters
params, labels = obs_params()
I, t, t_on, t_off = syn_current(duration=150, dt=dt, on_off=True)
x_o = syn_obs_data(I, dt, params)
prior = prior(params)


S = syn_obs_stats(I, params=params, dt=dt, t_on=t_on, t_off=t_off+1, n_summary=4,
              summary_stats=3, data=x_o)


M = DAPSimulator(I, dt, -75)
sum_stats = HodgkinHuxleyStatsSpikes(t_on, t_off, n_summary=4)


G = Default(model=M, prior=prior, summary=sum_stats)  # Generator
inf_snpe = SNPE(generator=G, n_components=2, n_hiddens=[10], obs=S)
# verbose=True, svi=False)
logs, tds, posteriors = inf_snpe.run(n_train=[n_samples], n_rounds=n_rounds,
                                     monitor=observables, proposal=prior,
                                     round_cl=1)


posterior_sampl = simulate_data_distr(posteriors[-1], M, sum_stats, n_samples=10)
print(posterior_sampl)

print('Saving Data')
sys.setrecursionlimit(10000)
# save(inf_snpe, directory + '/dap_snpe' + args.name)

save_pkl(I, directory + '/dap_I' + args.name)
save_pkl(dt, directory + '/dap_dt' + args.name)
save_pkl(t_on, directory + '/dap_t_on' + args.name)
save_pkl(t_off, directory + '/dap_t_off' + args.name)

save_pkl(M, directory + '/dap_model' + args.name)
save_pkl(sum_stats, directory + '/dap_stats' + args.name)
# save_pkl(S, directory + '/dap_stats' + args.name)
save_pkl(G, directory + '/dap_gen' + args.name)

save_pkl(logs, directory + '/dap_logs' + args.name)
save_pkl(tds, directory + '/dap_tds' + args.name)
save_pkl(posteriors, directory + '/dap_posteriors' + args.name)
save_pkl(prior, directory + '/dap_prior' + args.name)
save_pkl(params, directory + '/dap_params' + args.name)
save_pkl(labels, directory + '/dap_labels' + args.name)
