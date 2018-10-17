import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP


start = time.time()

gbar_kdr = 0.00313  # (S/cm2)
gbar_hcn = 5e-05    # (S/cm2)
gbar_nap = 0.01527  # (S/cm2)
gbar_nat = 0.142    # (S/cm2)

params = [gbar_kdr, gbar_hcn, gbar_nap, gbar_nat]


# stiumlation current
T = 300
dt = 0.01
l = T/dt
t = np.linspace(0, T, int(l))

I0 = np.zeros_like(t)
i_up = np.linspace(0,3.5,250)
i_down = np.linspace(3.5,0,250)
i_both = np.append(i_up, i_down)
I0[15000:15500] = i_both[:]

# define model
dap1 = DAP(-75, params)

# run model
# UDap, M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr = dap1.simulate(T, dt, I1)
UDap = dap1.simulate(dt, t, I0)

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

end = time.time()
print(end - start)

plt.show()
