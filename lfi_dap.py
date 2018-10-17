import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP, DAPSimulator
from DAPmodel import obs_params, syn_current, syn_obs_data, prior

from delfi.inference import SNPE
from delfi.summarystats import Identity
from delfi.utils.io import save, save_pkl
from delfi.generator import Default


n_rounds = 1
n_samples = 50

# picking experiments observables
observables = {'loss.lprobs', 'imputation_values', 'h1.mW', 'h1.mb', 'h2.mW',
               'h2.mb', 'weights.mW', 'weights.mb', 'means.mW0', 'means.mW1',
               'means.mb0', 'means.mb1', 'precisions.mW0', 'precisions.mW1',
               'precisions.mb0', 'precisions.mb1'}

# setting up fake parameters
params, labels = obs_params()
I, t = syn_current(duration=300, dt=0.01)

x_o = syn_obs_data(I, 0.01, params)

sum_stats = Identity()
prior = prior(params)

# LFI
M = DAPSimulator(I, 0.01, -75)

S = Identity()  # summary statistics
G = Default(model=M, prior=prior, summary=S)  # Generator
inf_snpe = SNPE(generator=G, n_components=2, n_hiddens=[10, 10], obs=sum_stats,
                verbose=True, svi=False)
logs, tds, posteriors = inf_snpe.run(n_train=[n_samples], n_rounds=n_rounds,
                                     monitor=observables, proposal=prior,
                                     round_cl=1)
