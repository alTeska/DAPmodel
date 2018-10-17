import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP_Model


start = time.time()
# stiumlation current
T = 600
dt = 0.01
l = T/dt
t = np.linspace(0, T, int(l))
I1 = np.zeros_like(t)
I1[25000:26000] = 3 #* 1e-3  # input should be in uA (nA * 1e-3)

I0 = np.zeros_like(t)
i_up = np.linspace(0,3.5,250)
i_down = np.linspace(3.5,0,250)
i_both = np.append(i_up, i_down)
I0[25000:25500] = i_both[:]

# define model
dap1 = DAP_Model()

# run model
# UDap, M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr = dap1.simulate(T, dt, I1)
UDap = dap1.simulate(T, dt, I0)

# I0 current
# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));
ax[0].grid()
ax[0].set_ylabel('V (mV)')
ax[0].set_xlabel('t (ms)')
ax[0].plot(t, UDap, label='membrane potential');
ax[1].plot(t, I0);

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


# I1 current
# UDap1, M_nap1, M_nat1, H_nap1, H_nat1, N_hcn1, N_kdr1 = dap1.simulate(T, dt, I1)
# UDap1 = dap1.simulate(T, dt, I1)

# plot voltage trace
# fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));
# ax[0].grid()
# ax[0].set_ylabel('V (mV)')
# ax[0].set_xlabel('t (ms)')
# ax[0].plot(t, UDap1, label='membrane potential');
# ax[1].plot(t, I1);

# plot of activation functions
# fig, ax = plt.subplots(ncols=1, figsize=(20, 10));
# ax.grid()
# ax.set_ylabel('x')
# ax.set_xlabel('t (ms)')
#
# ax.plot(t, M_nap1, label='M_nap');
# ax.plot(t, M_nat1, label='M_nat');
# ax.plot(t, H_nap1, label='H_nap');
# ax.plot(t, H_nat1, label='H_nat');
# ax.plot(t, N_kdr1, label='N_kdr');
# ax.plot(t, N_hcn1, label='N_hcn');
# plt.legend();

end = time.time()
# print(end - start)

plt.show()
