import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP
from DAPmodel import obs_params, syn_current
from lfimodels.hodgkinhuxley.HodgkinHuxleyBioPhys import HH

dt = 0.01

params, labels = obs_params()
I, t, t_on, t_off = syn_current(duration=150, dt=dt)


params = np.array([13, 16, 15])
# define model
dap1 = DAP(-75, params)
hh1 = HH(init=[-75], params=[[50, 5]])

# run model
Uhh = hh1.sim_time(dt, t, I*1e2)
UDap = dap1.simulate(dt, t, I)

print(Uhh.shape)
print(UDap.shape)

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));
ax[0].grid()
ax[0].set_ylabel('V (mV)')
ax[0].set_xlabel('t (ms)')
ax[0].plot(t, UDap, label='DAP');
ax[0].plot(t, Uhh, label='hh');
ax[0].legend()
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
