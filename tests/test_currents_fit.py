import time
import numpy as np
import matplotlib.pyplot as plt
from dap import DAPcython
from dap.utils import obs_params_gbar, syn_current
from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                    get_v_and_t_from_heka, shift_v_rest, get_i_inj_zap)



data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'  # best cell
protocol='IV'
ramp_amp=1


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
I, t_on, t_off = get_i_inj_from_function(protocol, [sweep_idx], t[-1], dt,
                                         return_discontinuities=False)
I = I[0]


plt.plot(I)
plt.plot(v)
plt.show()
