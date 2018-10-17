import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP
from DAPmodel import DAPSimulator


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
dap1 = DAPSimulator(I0, dt, -75)

# run model
stats = dap1.gen_single(params)
print(stats['data'])
print(stats['time'])

# plot voltage trace
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10));
ax[0].grid()
ax[0].set_ylabel('V (mV)')
ax[0].set_xlabel('t (ms)')
ax[0].plot(stats['time'][:30000], stats['data'][:30000], label='membrane potential');
ax[1].plot(t, I0);

end = time.time()
# print(end - start)

plt.show()
