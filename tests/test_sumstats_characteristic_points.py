import matplotlib.pyplot as plt

from dap.dap_sumstats_dict import DAPSummaryStatsDict
from dap.dap_sumstats import DAPSummaryStats
from dap.utils import (obs_params, syn_obs_data)

from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                        get_v_and_t_from_heka, shift_v_rest)

n_rounds = 1
n_samples = 10
dt = 0.01

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


# generate data format for SNPE
x_o =  {'data': v,
        'time': t,
        'dt': t[1]-t[0],
        'I': i_inj[0]}

sum_stats_dict = DAPSummaryStatsDict(t_on, t_off, n_summary=8)
sum_stats = DAPSummaryStats(t_on, t_off, n_summary=8)

print('summary stats:', sum_stats_dict.calc([x_o]))
print('summary stats A:', sum_stats.calc([x_o]), '\n')

# Print summary statistics for alternative values
params, labels = obs_params()
x_1 = syn_obs_data(i_inj[0], 0.01, params)

print('summary stats:', sum_stats.calc([x_1]))
print('summary stats A:', sum_stats.calc([x_1]), '\n')

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 10));
ax.grid()
ax.set_ylabel('V (mV)')
ax.set_xlabel('t (ms)')
ax.plot(x_1['time'], x_1['data'], label='DAP');
plt.plot()

# plt.show()
