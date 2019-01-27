import time
import numpy as np
import matplotlib.pyplot as plt
from dap import DAP, DAPBe, DAPExp, DAPFeExp
from dap.utils import obs_params, syn_current
from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                    get_v_and_t_from_heka, shift_v_rest, get_i_inj_zap)


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

    I, _ = get_i_inj_from_function(protocol, [sweep_idx], t[-1], dt,
                                              return_discontinuities=True)
    I = I[0]

    return I, t, dt, v


data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell

time_start = time.clock()

# load the data
I, t, dt, v = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)
params, labels = obs_params()

# define models
dap = DAP(-75, params)
dap_exp = DAPExp(-75, params)
dap_feexp = DAPFeExp(-75, params)
dap_be = DAPBe(-75, params)

# run model
# DAPdict = dap.simulate(dt, t, I, channels=True)
# DAPexpDict = dap_exp.simulate(dt, t, I, channels=True)
# DAPfexpDict = dap_feexp.simulate(dt, t, I, channels=True)
DAPbeDict = dap_be.simulate(dt, t, I, channels=True)

time_end = time.clock()
print('time elapsed:', time_end - time_start)

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));

ax[1].plot(t, DAPbeDict['U'], label='DapBe');
ax[1].plot(t, v, label='data');
ax[1].grid()
ax[1].legend()
ax[0].plot(t, I);


plt.show()
