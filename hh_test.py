import numpy as np
import matplotlib.pyplot as plt

from DAPmodel import Hodgkin_Huxley_Model


# stiumlation current
l = 100/0.01
t = np.linspace(0, 100, int(l))
I1 = np.repeat(0, int(l))

I1[1499:2000] = 3

# define model
h1 = Hodgkin_Huxley_Model(6.3)

# run model
U1, M1, N1, H1 = h1.simulate(100, 0.01, I1)


# plot results
fig, ax = plt.subplots(ncols=1, figsize=(20, 10));
ax.grid()
ax.set_ylabel('V (mV)')
ax.set_xlabel('t (ms)')
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 10));
ax.set_ylim(-20, 120)
ax.set_yticks(np.arange(-20, 120, 20));
ax.plot(t, U1, label='membrane potential');

fig, ax = plt.subplots(ncols=1, figsize=(20, 10));
ax.grid()
ax.set_ylabel('x')
ax.set_xlabel('t (ms)')

ax.plot(t, M1, label='M1');
ax.plot(t, H1, label='H1');
ax.plot(t, N1, label='N1');
plt.legend();

plt.show()
