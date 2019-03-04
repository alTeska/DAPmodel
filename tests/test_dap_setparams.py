import matplotlib.pyplot as plt
from dap import DAP, DAPBe
from dap import DAPcython
from dap.utils import obs_params, obs_params_gbar, syn_current


dt = 1e-2
params, labels = obs_params(reduced_model=True)
params, labels = obs_params_gbar(reduced_model=False)
I, t, t_on, t_off = syn_current(duration=120, dt=dt)

print(params)

# define models / check setters
dap = DAP(-75, params)
dap_back = DAPBe(-75, params)
dap_cython = DAPcython(-75, params, solver=1)
dap_cython_back = DAPcython(-75, params, solver=2)

# run models
U = dap.simulate(dt, t, I)
U_cython = dap_cython.simulate(dt, t, I)
U_back = dap_back.simulate(dt, t, I)
U_cython_back = dap_cython_back.simulate(dt, t, I)


# plot
fig, ax = plt.subplots(5, 1, figsize=(20, 10));
ax[0].plot(t, U, label='forward')
ax[1].plot(t, U_cython)
ax[2].plot(t, U_back, label='backward')
ax[3].plot(t, U_cython_back, label='backward_cython')

ax[4].plot(t, U, label='forward')
ax[4].plot(t, U_cython)
ax[4].plot(t, U_back, label='backward')
ax[4].plot(t, U_cython_back, label='backward_cython')
ax[4].legend()

ax[0].set_title('forward')
ax[1].set_title('forward_cython')
ax[2].set_title('backward')
ax[3].set_title('backward_cython')
ax[4].set_title('all')

plt.show()
