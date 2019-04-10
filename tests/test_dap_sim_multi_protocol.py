import numpy as np
import matplotlib.pyplot as plt
from dap.dap_sim_multi_protocol import DAPSimulatorMultiProtocol
from dap.utils import obs_params, load_current

np.set_printoptions(suppress=True, precision=2)


dt = 0.01
params, labels = obs_params()
params_list = [params]
data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell

# load the data
Ir, vr, tr, t_onr, t_offr, dtr = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)
Is, vs, ts, t_ons, t_offs, dts = load_current(data_dir, protocol='IV', ramp_amp=1)
I_all = [Ir, Is]
dt_all = [dtr, dts]

# define model
dap1 = DAPSimulatorMultiProtocol(I_all, dt_all, -75)

# run model
stats = dap1.gen_single(params, I_all[0], tr, dtr)
data_list = dap1.gen(params_list)


# plot inputs
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))
ax[0].plot(Ir)
ax[1].plot(Is)

# plot voltage traces
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))
ax[0].plot(data_list[0][0]['data'])
ax[1].plot(data_list[1][0]['data'])

plt.show()
