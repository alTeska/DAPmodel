import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP
from DAPmodel import obs_params, syn_current


params, labels = obs_params()
I, t = syn_current(duration=300, dt=0.01)

# define model
dap1 = DAP(-75, params)

# run model
# UDap, M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr = dap1.simulate(T, dt, I1)
UDap = dap1.simulate(0.01, t, I)

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));
ax[0].grid()
ax[0].set_ylabel('V (mV)')
ax[0].set_xlabel('t (ms)')
ax[0].plot(t, UDap, label='membrane potential');
ax[1].plot(t, I);

# plot of activation functions
# fig, ax = plt.subplots(ncols=1, figsize=(20, 10));
# ax.grid()
# ax.set_ylabel('x')
# ax.set_xlabel('t (ms)')
#
# ax.plot(t, M_nap, label='M_nap');
# ax.plot(t, M_nat, label='M_nat');
# ax.plot(t, H_nap, label='H_nap');
# ax.plot(t, H_nat, label='H_nat');
# ax.plot(t, N_kdr, label='N_kdr');
# ax.plot(t, N_hcn, label='N_hcn');
# plt.legend();

plt.show()
