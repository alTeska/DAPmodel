import numpy as np
import matplotlib.pyplot as plt

from dap import DAPcython
from dap.dap_sumstats_step_mom import DAPSummaryStatsStepMoments
from dap.utils import obs_params_gbar, syn_current
from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                        get_v_and_t_from_heka, shift_v_rest)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

n_rounds = 1
n_samples = 10
dt = 0.01


def load_current(data_dir, protocol='rampIV', ramp_amp=3.1):
    '''
    ramp_amp:  optimal=3.1, steps of 0.05 -0.15
    protocol: 'rampIV', 'IV', 'Zap20'

    '''
    v_shift = -16  # shift for accounting for the liquid junction potential

    if protocol == 'Zap20':
        sweep_idx = 0
    else:
        sweep_idx = get_sweep_index_for_amp(ramp_amp, protocol)

    v, t = get_v_and_t_from_heka(data_dir, protocol, sweep_idxs=[sweep_idx])
    v = shift_v_rest(v[0], v_shift)
    t = t[0]
    dt = t[1] - t[0]

    I, t_on, t_off = get_i_inj_from_function(protocol, [sweep_idx], t[-1], dt,
                                              return_discontinuities=False)

    I = I[0]

    return I, v, t, t_on, t_off, dt



dt = 1e-2
params, labels = obs_params_gbar()
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell


# load the data
I, v, t, t_on, t_off, dt = load_current(data_dir, protocol='IV', ramp_amp=1)

# generate syntetic data
params, labels = obs_params_gbar(reduced_model=True)

dap = DAPcython(-75, params)
U = dap.simulate(dt, t, I)

print(U.shape)
print(v.shape)
# generate data format for summary statistcs
x_o = {'data': v,
       'time': t,
       'dt': t[1]-t[0],
       'I': I}

x_1 = {'data': U.reshape(-1),
       'time': t,
       'dt': t[1]-t[0],
       'I': I}


# calcualte summary statistics
sum_stats = DAPSummaryStatsStepMoments(t_on, t_off, n_summary=17)

print('summary stats real:', sum_stats.calc([x_o]))
print('summary stats synt:', sum_stats.calc([x_1]))

# vizualize solution
fig, ax = plt.subplots(1,1, figsize=(10, 5))
ax.plot(t, I, label='I')
ax.plot(t, v, label='real')
ax.plot(t, U, label='synthetic')
ax.legend()
plt.show()
