import numpy as np
import matplotlib.pyplot as plt
from dap import DAPcython
from dap.dap_sumstats_step_mom import DAPSummaryStatsStepMoments
from dap.utils import obs_params_gbar, load_current
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell

# load the data
I, v, t, t_on, t_off, dt = load_current(data_dir, protocol='IV', ramp_amp=1)

# generate syntetic data
params, labels = obs_params_gbar(reduced_model=True)

dap = DAPcython(-75, params)
U = dap.simulate(dt, t, I)
print(U.shape)
print(v.shape)

# generate data format for summary statistcs
x_o = {'data': v,
       'time': t,
       'dt': t[1]-t[0],
       'I': I}

x_1 = {'data': U.reshape(-1),
       'time': t,
       'dt': t[1]-t[0],
       'I': I}


# calcualte summary statistics
sum_stats = DAPSummaryStatsStepMoments(t_on, t_off, n_summary=17)

print('summary stats real:', sum_stats.calc([x_o]))
print('summary stats synt:', sum_stats.calc([x_1]))
print('summary stats real:', np.size(sum_stats.calc([x_o])))
print('summary stats synt:', np.size(sum_stats.calc([x_1])))

# vizualize solution
fig, ax = plt.subplots(1,1, figsize=(10, 5))
ax.plot(t, I, label='I')
ax.plot(t, v, label='real')
ax.plot(t, U, label='synthetic')
ax.legend()
plt.show()
