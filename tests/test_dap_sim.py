
import matplotlib.pyplot as plt
from dap.dap_simulator import DAPSimulator
from dap.utils import obs_params_gbar, syn_current


params, labels = obs_params_gbar()
I, t, t_on, t_off = syn_current(duration=150, dt=0.01)

# define model
dap1 = DAPSimulator(I, 0.01, -75)

# run model
stats = dap1.gen_single(params)

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));
ax[0].grid()
ax[0].set_ylabel('V (mV)')
ax[0].set_xlabel('t (ms)')
ax[0].plot(stats['time'], stats['data'], label='DAP');
ax[0].legend()
ax[1].plot(t, I);


plt.show()
