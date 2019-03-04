import timeit
import numpy as np
import matplotlib.pyplot as plt

from dap import DAPcython
from dap.dap_sumstats_moments import DAPSummaryStatsMoments

from dap.utils import obs_params_gbar, syn_current
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

    I, t_on, t_off = get_i_inj_from_function(protocol, [sweep_idx], t[-1], dt,
                                              return_discontinuities=False)

    I = I[0]

    return I, v, t, t_on, t_off, dt



dt = 1e-2
params, labels = obs_params_gbar()
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell


# load the data
I, v, t, t_on, t_off, dt = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)
I_iv, v_iv, t_iv, t_on_iv, t_off_iv, dt_iv = load_current(data_dir, protocol='IV', ramp_amp=1)



# define and run the model
dap = DAPcython(-75, params)
U = dap.simulate(dt, t, I)
U_iv = dap.simulate(dt_iv, t_iv, I_iv)

print("cython:", timeit.timeit(lambda: dap.simulate(dt, t, I), number=int(1)))
print("cython:", timeit.timeit(lambda: dap.simulate(dt_iv, t_iv, I_iv), number=int(1)))

print(dt, dt_iv)



fig, ax = plt.subplots(2, 1, figsize=(10,20))
ax[0].plot(t, I)
ax[0].plot(t, v)
ax[0].plot(t, U)
ax[1].plot(t_iv, I_iv)
ax[1].plot(t_iv, v_iv)
ax[1].plot(t_iv, U_iv)

plt.show()
