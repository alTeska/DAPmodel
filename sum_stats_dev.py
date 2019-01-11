import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin, argrelmax

from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                        get_v_and_t_from_heka, shift_v_rest, get_i_inj_zap)

from dap import DAPBe, DAPcython
from dap.dap_sumstats_dict import DAPSummaryStatsDict
from dap.dap_sumstats import DAPSummaryStats
from dap.dap_simulator import DAPSimulator

from dap.analyze_APs import get_spike_characteristics,  get_spike_characteristics_dict
from dap.utils import obs_params, syn_current
from dap.utils import (obs_params, syn_current, syn_obs_data, prior,
                       syn_obs_stats)

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


def sumstats(v, t, t_on, t_off):
    # RMSE
    # n = len(v0)
    # rmse = np.linalg.norm(v - v0) / np.sqrt(n)

    rest_pot = np.mean(v[t < t_on])

    # more then one AP:
    X = np.shape(np.where(v > 30))[1]


    # hyperpolarization after DAP
    mAHP_idx = np.argmin(v)
    mAHP = v[mAHP_idx]

    # Action potential
    threshold = -30
    AP_onsets = np.where(v > threshold)[0]
    AP_start = AP_onsets[0]
    AP_end = AP_onsets[-1]
    AP_max_idx = AP_start + np.argmax(v[AP_start:AP_end])
    AP_max = v[AP_max_idx]
    AP_amp = AP_max - rest_pot

    # AP width
    AP_onsets_half_max = np.where(v > (AP_max+rest_pot)/2)[0]
    AP_width = t[AP_onsets_half_max[-1]] - t[AP_onsets_half_max[0]]

    # DAP: fAHP
    v_dap = v[AP_max_idx:]
    fAHP_idx = argrelmin(v[AP_max_idx:])[0][0] + AP_max_idx
    fAHP = v[fAHP_idx]

    # DAP amplitude
    DAP_max_idx = argrelmax(v_dap)[0][1] + AP_max_idx
    DAP_max = v[DAP_max_idx]
    DAP_amp = DAP_max - rest_pot

    DAP_deflection = DAP_amp - (fAHP - rest_pot)
    DAP_time = t[DAP_max_idx] - t[AP_max_idx]   # Time between AP and DAP maximum

    # Width of DAP: between fAHP and halfsize of fAHP after DAP max
    vnorm = v[DAP_max_idx:] - rest_pot


    half_max = np.where((abs(vnorm) < abs(fAHP - rest_pot)/2))[0]

    DAP_width_idx = DAP_max_idx + half_max[0]
    DAP_width = (DAP_width_idx - fAHP_idx) * dt

    # print('\n', 'LOCAL')
    # print('rest_pot', rest_pot)
    # print('AP_amp', AP_amp)
    # print('AP_width', AP_width)
    # print('fAHP', fAHP)
    # print('DAP_amp', DAP_amp)
    # print('DAP_width', DAP_width)
    # print('DAP_deflection', DAP_deflection)
    # print('DAP_time', DAP_time)
    # print('mAHP', mAHP)

    sum_stats_vec = np.array([
                    # rmse,
                    rest_pot,
                    AP_amp,
                    AP_width,
                    fAHP,
                    DAP_amp,
                    DAP_width,
                    DAP_deflection,
                    DAP_time,
                    mAHP,
                    ])


    # return sum_stats_vec, DAP_max, DAP_max_idx, AP_max, AP_max_idx, fAHP, fAHP_idx, rest_pot, mAHP_idx, mAHP, DAP_width_idx
    return sum_stats_vec, DAP_max, DAP_max_idx, AP_max, AP_max_idx, fAHP, fAHP_idx, rest_pot, mAHP_idx, mAHP, DAP_width_idx





# gbar_nap       [0   ; 0.5]   ( 0.01527)
# nap_m_vs       [1   ; 30 ]   ( 16.11  )
dt = 1e-2
params, labels = obs_params()
params_test = np.array([0.015, 16])  # for stability test

data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell

# load the data
I, t, t_on, t_off, dt = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)

# define models
dap = DAPcython(-75, params)
m = DAPSimulator(I=I, dt=dt, V0=-75)
s = DAPSummaryStats(t_on, t_off, n_summary=9)


# run models
U = dap.simulate(dt, t, I)
data = m.gen_single(params_test)
statistics, stats_idx = s.calc([data])

print(statistics)
print(stats_idx)
AP_max_idx = stats_idx[0][0]
fAHP_idx = stats_idx[0][1]
mAHP_idx = stats_idx[0][2]
DAP_max_idx = stats_idx[0][3]
DAP_width_idx = stats_idx[0][4]

rest_pot = statistics[0][0]
AP_amp = statistics[0][1]
AP_width = statistics[0][2]
fAHP = statistics[0][3]
DAP_amp = statistics[0][4]
DAP_width = statistics[0][5]
DAP_deflection = statistics[0][6]
DAP_time = statistics[0][7]
mAHP = statistics[0][8]
# U = data['data'].transpose()

# t = data['time']

# calculate local stats
# sum_stats_vec, DAP_max, DAP_max_idx, AP_max, AP_max_idx, fAHP, fAHP_idx, rest_pot, mAHP_idx, mAHP, DAP_width_idx = sumstats(U, t, t_on, t_off)

# print('\n diff:',statistics-sum_stats_vec)

# Plot the results
plt.figure()
plt.plot(t, I)
plt.plot(t, U.transpose()[0], label='U_params')
plt.plot(t, data['data'], label='U_testing')
plt.legend()

plt.plot(t[DAP_max_idx], DAP_max, '*')
plt.plot(t[AP_max_idx], AP_max, '*')
plt.plot(t[fAHP_idx], fAHP, '*')
plt.plot(t[mAHP_idx], mAHP, '*')
plt.plot([t[DAP_max_idx], t[AP_max_idx]], [-70, -70])
plt.plot(t[DAP_width_idx], U[DAP_width_idx], 's')
plt.plot([t[fAHP_idx], t[DAP_width_idx]], [U[fAHP_idx], U[DAP_width_idx]], '--')

plt.show()
