import numpy as np
import matplotlib.pyplot as plt

from dap.dap_sumstats_moments import DAPSummaryStatsMoments
from dap import DAPcython
from dap.utils import obs_params_gbar, syn_current

from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                        get_v_and_t_from_heka, shift_v_rest)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

n_rounds = 1
n_samples = 10
dt = 0.01

# load real data
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
protocol = 'rampIV' # 'IV' # 'rampIV' # 'Zap20'
ramp_amp = 3.1 # steps of 0.05 -0.15
v_shift = -16  # shift for accounting for the liquid junction potential

sweep_idx = get_sweep_index_for_amp(ramp_amp, protocol)

v, time = get_v_and_t_from_heka(data_dir, protocol, sweep_idxs=[sweep_idx])
v = shift_v_rest(v[0], v_shift)
time = time[0]
i_inj, t_on, t_off = get_i_inj_from_function(protocol, [sweep_idx], time[-1], time[1]-time[0],
                                             return_discontinuities=False)

# generate syntetic data
params, labels = obs_params_gbar(reduced_model=True)

dap = DAPcython(-75, params)
U = dap.simulate(dt, time, i_inj[0])

print(U.shape)
print(v.shape)
# generate data format for summary statistcs
x_o = {'data': v,
       'time': time,
       'dt': time[1]-time[0],
       'I': i_inj[0]}

x_1 = {'data': U.reshape(-1),
       'time': time,
       'dt': time[1]-time[0],
       'I': i_inj[0]}


# calcualte summary statistics
sum_stats_mom = DAPSummaryStatsMoments(t_on, t_off, n_summary=17)

print('moments summary stats real:', sum_stats_mom.calc([x_o]))
print('moments summary stats synt:', sum_stats_mom.calc([x_1]))


# vizualize solution
fig, ax = plt.subplots(1,1, figsize=(10, 5))
ax.plot(time, i_inj[0], label='I')
ax.plot(time, v, label='real')
ax.plot(time, U, label='synthetic')
ax.legend()
plt.show()
