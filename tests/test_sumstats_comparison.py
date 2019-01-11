import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin, argrelmax

from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                    get_v_and_t_from_heka, shift_v_rest, get_i_inj_zap)

from dap import DAP, DAPBe, DAPExp, DAPFeExp
from dap.analyze_APs import get_spike_characteristics,  get_spike_characteristics_dict
from dap.utils import obs_params, syn_current


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
    fAHP_idx = argrelmin(v)[0][1]
    fAHP = v[fAHP_idx] - rest_pot

    # DAP amplitude
    DAP_max_idx = argrelmax(v)[0][1]
    DAP_max = v[DAP_max_idx]
    DAP_amp = DAP_max - rest_pot

    DAP_deflection = DAP_amp - fAHP
    DAP_time = t[DAP_max_idx] - t[AP_max_idx] # Time between AP and DAP maximum

    # Width of DAP: between fAHP and halfsize of fAHP after DAP max
    vnorm = v[fAHP_idx:-1] - rest_pot
    half_max = np.where((vnorm < fAHP/2))[0]
    DAP_width = half_max[0] * dt

    sum_stats_vec = np.array([
                    # rest_pot,
                    # rmse,
                    AP_amp[0],
                    AP_width,
                    DAP_amp[0],
                    DAP_width,
                    DAP_deflection[0],
                    DAP_time,
                    # rest_pot_std, # should it be included?
                    ])

    return sum_stats_vec, DAP_max, DAP_max_idx, AP_max, AP_max_idx, fAHP, fAHP_idx, rest_pot



def sum_stats_analyze(v, t, t_on, t_off):
    return_characteristics = ['AP_amp', 'AP_width', 'DAP_amp', 'DAP_width',
                              'DAP_deflection', 'DAP_time',
                              'AP_max_idx','DAP_max_idx', 'v_rest']
    spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)  # standard parameters to use

    characteristics = get_spike_characteristics(v, t, return_characteristics,
                                                v_rest=v[0], std_idx_times=(0, 1)
                                                ,**spike_characteristics_dict)

    return characteristics


dt = 1e-2
params, labels = obs_params()
params = np.array([.5, 0.4])  # for stability test
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell

# load the data

# I, t, t_on, t_off = syn_current(duration=150, dt=dt)
I, t, t_on, t_off, dt = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)


# define models
dap = DAP(-75, params)
dap_be = DAPBe(-75, params)

# run models
U = dap_be.simulate(dt, t, I)
# U0 = dap.simulate(dt, t, I)

sumstats_hand, DAP_max, DAP_max_idx, AP_max, AP_max_idx, fAHP, fAHP_idx, rest_pot = sumstats(U, t, t_on, t_off)
print(sumstats_hand, '\n')

# characteristics = sum_stats_analyze(U.transpose()[0], t, t_on, t_off)
# print(
#     characteristics['AP_amp'],
#     characteristics['AP_width'],
#     characteristics['DAP_amp'],
#     characteristics['DAP_width'],
#     characteristics['DAP_deflection'],
#     characteristics['DAP_time'], '\n'
#     )

plt.figure()
plt.plot(t, I)
plt.plot(t, U.transpose()[0])

plt.plot(t[DAP_max_idx], DAP_max, '*')
plt.plot(t[AP_max_idx], AP_max, '*')
plt.plot(t[fAHP_idx], fAHP+rest_pot, '*')

# plt.plot(t[characteristics['DAP_max_idx']], characteristics['DAP_amp'] + characteristics['v_rest'], 'o')
# plt.plot(t[characteristics['AP_max_idx']], characteristics['AP_amp'] + characteristics['v_rest'], 'o')

plt.show()
