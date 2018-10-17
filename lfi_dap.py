import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP, DAPSimulator
from DAPmodel import obs_params, syn_current

from delfi.inference import SNPE
from delfi.summarystats import Identity
from delfi.utils.io import save, save_pkl
from delfi.generator import Default



# picking experiments observables
observables = {'loss.lprobs', 'imputation_values', 'h1.mW', 'h1.mb', 'h2.mW',
               'h2.mb', 'weights.mW', 'weights.mb', 'means.mW0', 'means.mW1',
               'means.mb0', 'means.mb1', 'precisions.mW0', 'precisions.mW1',
               'precisions.mb0', 'precisions.mb1'}

# setting up fake parameters
params, labels = obs_params()
I, dt = syn_current()

# x_o = utils.syn_obs_data(I, dt, params)    # data
# sum_stats = utils.syn_obs_stats(I, params, dt, t_on, t_off, n_summary=4)
# prior = utils.prior(params)
#
# # LFI
# S = Identity()  # summary statistics
# G = Default(model=M, prior=prior, summary=S)  # Generator
# inf_snpe = SNPE(generator=G, n_components=2, n_hiddens=[10, 10], obs=sum_stats,
#                 verbose=True, svi=False)
# logs, tds, posteriors = inf_snpe.run(n_train=[n_samples], n_rounds=n_rounds,
#                                      monitor=observables, proposal=prior,
#                                      round_cl=1)
