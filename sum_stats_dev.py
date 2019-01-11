import numpy as np
import matplotlib.pyplot as plt
from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                        get_v_and_t_from_heka, shift_v_rest, get_i_inj_zap)
from dap import DAPcython
from dap.utils import obs_params
from dap.dap_simulator import DAPSimulator
from dap.dap_sumstats import DAPSummaryStats
# from dap.dap_sumstats_dict import DAPSummaryStatsDict


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

    return I, t, t_on, t_off, dt


# gbar_nap       [0   ; 0.5]   ( 0.01527)
# nap_m_vs       [1   ; 30 ]   ( 16.11  )
dt = 1e-2
params, labels = obs_params()
params_test = np.array([0.01527, 16.11])  # for stability test
params_test = np.array([0.05, 19.53])  # for stability test

data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell

# load the data
I, t, t_on, t_off, dt = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)

# define models
dap = DAPcython(-75, params)
m = DAPSimulator(I=I, dt=dt, V0=-75)
s = DAPSummaryStats(t_on, t_off, n_summary=9)

# run models
V = dap.simulate(dt, t, I)
data = m.gen_single(params_test)
statistics = s.calc([data])
# statistics, stats_idx = s.calc([data])

U = data['data']
# AP_max_idx = stats_idx[0][0]
# fAHP_idx = stats_idx[0][1]
# mAHP_idx = stats_idx[0][2]
# DAP_max_idx = stats_idx[0][3]
# DAP_width_idx = stats_idx[0][4]

rest_pot = statistics[0][0]
AP_amp = statistics[0][1]
AP_width = statistics[0][2]
fAHP = statistics[0][3]
DAP_amp = statistics[0][4]
DAP_width = statistics[0][5]
DAP_deflection = statistics[0][6]
DAP_time = statistics[0][7]
mAHP = statistics[0][8]

DAP_max = DAP_amp + rest_pot
AP_max = AP_amp + rest_pot


print('rest_pot', rest_pot)
print('AP_amp', AP_amp)
print('AP_width', AP_width)
print('fAHP', fAHP)
print('DAP_amp', DAP_amp)
print('DAP_width', DAP_width)
print('DAP_deflection', DAP_deflection)
print('DAP_time', DAP_time)
print('mAHP', mAHP)


# Plot the results
plt.figure(figsize=(15,10))
plt.ylim(-100, 100)
plt.plot(t, I)
plt.plot(t, V.transpose()[0], label='U_goal')
plt.plot(t, U, label='U_param')
plt.legend()
#
# plt.plot(t[DAP_max_idx], DAP_max, '*')
# plt.plot(t[AP_max_idx], AP_max, '*')
# plt.plot(t[fAHP_idx], fAHP, '*')
# plt.plot(t[mAHP_idx], mAHP, '*')
# plt.plot([t[DAP_max_idx], t[AP_max_idx]], [-70, -70], '--')
# plt.plot(t[DAP_width_idx], U[DAP_width_idx], 's')
# plt.plot([t[fAHP_idx], t[DAP_width_idx]], [U[fAHP_idx], U[DAP_width_idx]], '--')

plt.show()
