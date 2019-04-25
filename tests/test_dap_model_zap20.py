import time
import numpy as np
import matplotlib.pyplot as plt
from dap import DAPcython
from dap.utils import obs_params_gbar, load_current

params, labels = obs_params_gbar()

data_dir = '/home/alteska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'  # best cell
# data_dir = '/home/alteska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell


# load the data TODO: load actual Zap20 with CORRECT ramp_amp
I, v, t, t_on, t_off, dt = load_current(data_dir, protocol='IV', ramp_amp=-0.15)

time_start = time.clock()

# define models
dap = DAPcython(-75, params)

# run model
DAPdict = dap.simulate(dt, t, I, channels=True)

time_end = time.clock()
print('time elapsed:', time_end - time_start)

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 10));
ax.plot(t, DAPdict['U'], label='model');
ax.plot(t, v, label='data');
ax.grid()
ax.plot(t, I);
ax.legend()
plt.show()

# plot activation functions
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 10));
ax.plot(t, DAPdict['M_nap'], label='M_nap');
ax.plot(t, DAPdict['M_nat'], label='M_nat');
ax.plot(t, DAPdict['H_nap'], label='H_nap');
ax.plot(t, DAPdict['H_nat'], label='H_nat');
ax.plot(t, DAPdict['N_hcn'], label='N_hcn');
ax.plot(t, DAPdict['N_kdr'], label='N_kdr');

ax.legend()
