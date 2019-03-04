from dap.utils import load_current
from dap.analyze_APs import (get_spike_characteristics, get_spike_characteristics_dict,
                                  check_measures)

data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell

# load the data
i_inj, v, t, t_on, t_off, dt = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)



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
