import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP
from DAPmodel import obs_params, syn_current
from cell_fitting.read_heka import get_sweep_index_for_amp, get_i_inj_from_function, get_v_and_t_from_heka, shift_v_rest


# load the data
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
protocol = 'rampIV' # 'IV' # 'rampIV' # 'Zap20'
ramp_amp = 3.1 # steps of 0.05 -0.15
v_shift = -16  # shift for accounting for the liquid junction potential

if protocol == 'Zap20':
    sweep_idx = 0
else:
    sweep_idx = get_sweep_index_for_amp(ramp_amp, protocol)

v, t = get_v_and_t_from_heka(data_dir, protocol, sweep_idxs=[sweep_idx])
v = shift_v_rest(v[0], v_shift)
t = t[0]

i_inj, t_on, t_off = get_i_inj_from_function(protocol, [sweep_idx], t[-1], t[1]-t[0],
                                             return_discontinuities=False)
dt = t[1] - t[0]
i_inj = i_inj[0]

params, labels = obs_params()

# define model
dap1 = DAP(-75, params)

# run model
UDap = dap1.simulate(dt, t, i_inj)

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));
ax[0].grid()
ax[0].set_ylabel('V (mV)')
ax[0].set_xlabel('t (ms)')
ax[0].plot(t, UDap, label='DAP');
ax[1].plot(t, i_inj);

plt.show()
