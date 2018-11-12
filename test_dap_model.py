import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP, DAPBe, DAPExp, DAPExp2
from DAPmodel import obs_params, syn_current

dt = 0.001

params, labels = obs_params()

# params = np.array([2, 0.4])  # for stability test
I, t, t_on, t_off = syn_current(duration=150, dt=dt)

# define model
dap = DAP(-75, params)
dap_exp = DAPExp(-75, params)
dap_exp2 = DAPExp2(-75, params)
# dap_be = DAPBe(-75, params)

# run model
UDap, M_nap, M_nat, H_nap = dap.simulate(dt, t, I)
UDapExp, M_nap_exp, M_nat_exp, H_nap_exp = dap_exp.simulate(dt, t, I)
UDapExp2, M_nap_exp2, M_nat_exp2, H_nap_exp2 = dap_exp2.simulate(dt, t, I)
# UDapBe = dap_be.simulate(dt, t, I)

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(20, 10));
ax[0].set_ylabel('V (mV)')
ax[0].set_xlabel('t (ms)')
ax[0].plot(t, UDap, label='DAP');
ax[0].set_title('Forward Euler')
ax[0].grid()

ax[1].plot(t, UDapExp, label='DAPExp');
ax[1].set_title('Exp Euler')
ax[1].grid()

ax[2].plot(t, UDapExp2, label='DAPExp2');
ax[2].set_title('Exp + Forward Euler')

ax[2].grid()
ax[3].plot(t, I);


# plot activation functions
fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(20, 10));
ax[0].plot(t, M_nap, label='M_nap');
ax[0].plot(t, M_nat, label='M_nat');
ax[0].plot(t, H_nap, label='H_nap');
ax[0].set_title('Forward Euler')
ax[0].legend()

ax[1].plot(t, M_nap_exp, label='M_nap_exp');
ax[1].plot(t, M_nat_exp, label='M_nat_exp');
ax[1].plot(t, H_nap_exp, label='H_nap_exp');
ax[1].set_title('Exp Euler')
ax[1].legend()

ax[2].plot(t, M_nap_exp2, label='M_nap_exp2');
ax[2].plot(t, M_nat_exp2, label='M_nat_exp2');
ax[2].plot(t, H_nap_exp2, label='H_nap_exp2');
ax[2].set_title('Exp + Forward Euler')
ax[2].legend()
plt.legend();


plt.show()
