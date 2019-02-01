import time
import numpy as np
import matplotlib.pyplot as plt
from dap import DAPcython
from dap.utils import obs_params_gbar, syn_current
from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                    get_v_and_t_from_heka, shift_v_rest, get_i_inj_zap)

# from test_dap_model_iinj import load_current

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

    # dt = 0.1
    I, _ = get_i_inj_from_function(protocol, [sweep_idx], t[-1], dt,
                                              return_discontinuities=True)
    I = I[0]

    return I, v, t, dt

data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'  # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell

time_start = time.clock()

# load the data
I, v, t, dt = load_current(data_dir, protocol='IV', ramp_amp=1)
print(np.shape(I))

params, labels = obs_params_gbar()
# params *=10
# params = np.array([0.159, 1.039])

plt.plot(t,I)
plt.plot(t,v)
plt.show()

# define models
dap = DAPcython(-75, params)

# run model
DAPdict = dap.simulate(dt, t, I, channels=True)

time_end = time.clock()
print('time elapsed:', time_end - time_start)

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 10));
ax.plot(t, DAPdict['U'], label='model');
ax.plot(t, v, label='data');
ax.grid()
ax.plot(t, I);
ax.legend()
plt.show()

# plot activation functions
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 10));
# ax.plot(t, DAPdict['M_nap'], label='M_nap');
# ax.plot(t, DAPdict['M_nat'], label='M_nat');
# ax.plot(t, DAPdict['H_nap'], label='H_nap');
# ax.plot(t, DAPdict['H_nat'], label='H_nat');
# ax.plot(t, DAPdict['N_hcn'], label='N_hcn');
# ax.plot(t, DAPdict['N_kdr'], label='N_kdr');
#
# ax.legend()
