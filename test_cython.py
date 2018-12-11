import timeit
import matplotlib.pyplot as plt

from DAPmodel import DAP, DAPBe
from DAPmodel.DAPcython import DAPcython
from DAPmodel.utils import obs_params, syn_current


dt = 1e-2
params, labels = obs_params()

I, t, t_on, t_off = syn_current(duration=120, dt=dt)

# define models
dap = DAP(-75, params)
dap_back = DAPBe(-75, params)
dap_cython = DAPcython(-75, params, solver=1)
dap_cython_back = DAPcython(-75, params, solver=2)

# run models
U = dap.simulate(dt, t, I)
U_cython = dap_cython.simulate(dt, t, I)
U_back = dap_back.simulate(dt, t, I)
U_cython_back = dap_cython_back.simulate(dt, t, I)

print('\n', 'forward:')
print("cython:", timeit.timeit(lambda: dap_cython.simulate(dt, t, I), number=int(4)))
print("python:", timeit.timeit(lambda: dap.simulate(dt, t, I)       , number=int(4)))


print('\n', 'backward:')
print("cython:", timeit.timeit(lambda: dap_cython_back.simulate(dt, t, I), number=int(4)))
print("python:", timeit.timeit(lambda: dap_back.simulate(dt, t, I)       , number=int(4)))

fig, ax = plt.subplots(2, 1, figsize=(20, 10));
ax[0].plot(t, U, label='forward')
ax[0].plot(t, U_cython, label='forward_cython')
ax[0].legend()
ax[1].plot(t, U_back, label='backward')
ax[1].plot(t, U_cython_back, label='backward_cython')
ax[1].legend()

plt.show()
