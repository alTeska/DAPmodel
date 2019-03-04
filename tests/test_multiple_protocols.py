import timeit
import matplotlib.pyplot as plt
from dap import DAPcython
from dap.utils import load_current, obs_params_gbar

params, labels = obs_params_gbar()
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell

# load the data
I, v, t, t_on, t_off, dt = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)
I_iv, v_iv, t_iv, t_on_iv, t_off_iv, dt_iv = load_current(data_dir, protocol='IV', ramp_amp=1)



# define and run the model
dap = DAPcython(-75, params)
U = dap.simulate(dt, t, I)
U_iv = dap.simulate(dt_iv, t_iv, I_iv)

print("cython:", timeit.timeit(lambda: dap.simulate(dt, t, I), number=int(1)))
print("cython:", timeit.timeit(lambda: dap.simulate(dt_iv, t_iv, I_iv), number=int(1)))

print(dt, dt_iv)

fig, ax = plt.subplots(2, 1, figsize=(10,20))
ax[0].plot(t, I)
ax[0].plot(t, v)
ax[0].plot(t, U)
ax[1].plot(t_iv, I_iv)
ax[1].plot(t_iv, v_iv)
ax[1].plot(t_iv, U_iv)

plt.show()
