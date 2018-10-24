import matplotlib.pyplot as plt
from cell_fitting.read_heka import get_sweep_index_for_amp, get_i_inj_from_function, get_v_and_t_from_heka, shift_v_rest

# parameters
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'

# load
ramp_amp = 3.1
v_shift = -16  # shift for accounting for the liquid junction potential
sweep_idx = get_sweep_index_for_amp(ramp_amp, 'rampIV')
v, t = get_v_and_t_from_heka(data_dir, 'rampIV', sweep_idxs=[sweep_idx])
v = shift_v_rest(v[0], v_shift)
t = t[0]
i_inj = get_i_inj_from_function('rampIV', [sweep_idx], t[-1], t[1]-t[0])[0]

# plot
plt.figure()
plt.plot(t, v, 'k', label='Exp. Data')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.tight_layout()
plt.show()
