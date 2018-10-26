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
from lfimodels.hodgkinhuxley.HodgkinHuxleyStatsSpikes_mf import HodgkinHuxleyStatsSpikes_mf

from cell_fitting.read_heka import get_sweep_index_for_amp, get_i_inj_from_function, get_v_and_t_from_heka, shift_v_rest


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

# load the data
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
protocol = 'rampIV' # 'IV' # 'rampIV' # 'Zap20'
ramp_amp = 3.5 # steps of 0.05 -0.15
v_shift = -16  # shift for accounting for the liquid junction potential

if protocol == 'Zap20':
    sweep_idx = 0
else:
    sweep_idx = get_sweep_index_for_amp(ramp_amp, protocol)

v, t = get_v_and_t_from_heka(data_dir, protocol, sweep_idxs=[sweep_idx])
v = shift_v_rest(v[0], v_shift)
t = t[0]
i_inj, t_on, t_off = get_i_inj_from_function(protocol, [sweep_idx], t[-1], t[1]-t[0], return_discontinuities=False)

# generate data format for SNPE
x_o =  {'data': v,
        'time': t,
        'dt': t[1]-t[0],
        'I': i_inj[0]}


# picking experiments observables
observables = {'loss.lprobs', 'imputation_values', 'h1.mW', 'h1.mb', 'h2.mW',
               'h2.mb', 'weights.mW', 'weights.mb', 'means.mW0', 'means.mW1',
               'means.mb0', 'means.mb1', 'precisions.mW0', 'precisions.mW1',
               'precisions.mb0', 'precisions.mb1'}

# setting up parameters

# _, labels = obs_params()
#'nap_m_tau_max': 15.332,   # ms
#'nap_m_vs': 16.11,         # mV
#'nap_h_tau_max': 13.659,   # ms


labels = ['nap_m_tau_max', 'nap_m_vs', 'nap_h_tau_max']
params = np.array([15, 16, 13])

print(params)
prior = prior(params)


S = syn_obs_stats(x_o['I'], params=params, dt=x_o['dt'], t_on=t_on, t_off=t_off, n_summary=3,
           summary_stats=2, data=x_o)

M = DAPSimulator(x_o['I'], x_o['dt'], -75)
sum_stats = HodgkinHuxleyStatsSpikes(t_on, t_off, n_summary=3)


G = Default(model=M, prior=prior, summary=sum_stats)  # Generator
inf_snpe = SNPE(generator=G, n_components=2, n_hiddens=[10], obs=S)
logs, tds, posteriors = inf_snpe.run(n_train=[n_samples], n_rounds=n_rounds,
                                     monitor=observables, round_cl=1)


posterior_sampl = simulate_data_distr(posteriors[-1], M, sum_stats, n_samples=10)
print(posterior_sampl)

print('Saving Data')
sys.setrecursionlimit(10000)
# save(inf_snpe, directory + '/dap_snpe' + args.name)

save_pkl(x_o['I'], directory + '/dap_I' + args.name)
save_pkl(x_o['dt'], directory + '/dap_dt' + args.name)
save_pkl(t_on, directory + '/dap_t_on' + args.name)
save_pkl(t_off, directory + '/dap_t_off' + args.name)

save_pkl(M, directory + '/dap_model' + args.name)
save_pkl(sum_stats, directory + '/dap_stats' + args.name)
save_pkl(S, directory + '/dap_stats_data' + args.name)
save_pkl(G, directory + '/dap_gen' + args.name)

save_pkl(logs, directory + '/dap_logs' + args.name)
save_pkl(tds, directory + '/dap_tds' + args.name)
save_pkl(posteriors, directory + '/dap_posteriors' + args.name)
save_pkl(prior, directory + '/dap_prior' + args.name)
save_pkl(params, directory + '/dap_params' + args.name)
save_pkl(labels, directory + '/dap_labels' + args.name)