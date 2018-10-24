import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAPSimulator
from DAPmodel import obs_params, syn_current
from lfimodels.hodgkinhuxley.HodgkinHuxley import HodgkinHuxley
from lfimodels.hodgkinhuxley.HodgkinHuxleyBioPhys import HH


params, labels = obs_params()
I, t, t_on, t_off = syn_current(duration=150, dt=0.01)

# define model
dap1 = DAPSimulator(I, 0.01, -75)
hh_sim = HodgkinHuxley(I*1e2, 0.01, -75)

# run model
stats = dap1.gen_single(params)
stats_hh = hh_sim.gen_single(params=[50, 5])

print(np.shape(stats['time']))
print(np.shape(stats_hh['time']))

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));
ax[0].grid()
ax[0].set_ylabel('V (mV)')
ax[0].set_xlabel('t (ms)')
ax[0].plot(stats['time'], stats['data'], label='membrane potential');
ax[0].plot(stats_hh['time'], stats_hh['data'], label='membrane potential');
ax[1].plot(t, I);

end = time.time()
# print(end - start)

plt.show()
