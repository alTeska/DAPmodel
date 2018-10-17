import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAPSimulator
from DAPmodel import obs_params, syn_current


params, labels = obs_params()
I, t = syn_current(duration=300, dt=0.01)

# define model
dap1 = DAPSimulator(I, 0.01, -75)

# run model
stats = dap1.gen_single(params)

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));
ax[0].grid()
ax[0].set_ylabel('V (mV)')
ax[0].set_xlabel('t (ms)')
ax[0].plot(stats['time'][:30000], stats['data'][:30000], label='membrane potential');
ax[1].plot(t, I);

end = time.time()
# print(end - start)

plt.show()
