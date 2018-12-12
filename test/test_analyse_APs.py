from cell_fitting.read_heka import (get_sweep_index_for_amp, get_i_inj_from_function,
                                    get_v_and_t_from_heka, shift_v_rest)
from DAPmodel.analyze_APs import (get_spike_characteristics, get_spike_characteristics_dict,
                                  check_measures)



# parameters
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'

# load
protocol = 'rampIV' #'IV'
amp = 3.1
v_shift = -16  # shift for accounting for the liquid junction potential
if protocol == 'Zap20':
    sweep_idx = 0
else:
    sweep_idx = get_sweep_index_for_amp(amp, protocol)
v, t = get_v_and_t_from_heka(data_dir, protocol, sweep_idxs=[sweep_idx])
v = shift_v_rest(v[0], v_shift)
t = t[0]
i_inj = get_i_inj_from_function(protocol, [sweep_idx], t[-1], t[1]-t[0])[0]

# extract AP/DAP characteristics
return_characteristics = ['AP_amp', 'AP_width', 'DAP_amp', 'DAP_width',
                          'DAP_deflection', 'DAP_time']
get_spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)  # standard parameters to use
# AP_amp, AP_width, DAP_amp, DAP_width, DAP_deflection, DAP_time = get_spike_characteristics(v, t, return_characteristics,
                                                                                 # v_rest=v[0], std_idx_times=(0, 1),
                                                                                 # check=True, **get_spike_characteristics_dict)

characteristics = get_spike_characteristics(v, t, return_characteristics,
                  v_rest=v[0], std_idx_times=(0, 1),
                  check=True, **get_spike_characteristics_dict)

check_measures(v, t, characteristics)

print(
    characteristics['AP_amp'],
    characteristics['AP_width'],
    characteristics['DAP_amp'],
    characteristics['DAP_width'],
    characteristics['DAP_deflection'],
    characteristics['DAP_time']
    )
