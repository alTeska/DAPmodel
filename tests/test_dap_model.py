import time
import numpy as np
import matplotlib.pyplot as plt

from dap import DAP, DAPBe, DAPExp, DAPFeExp
from dap.utils import obs_params, syn_current
from dap.dap_sumstats_dict import DAPSummaryStatsDict
from dap.dap_sumstats import DAPSummaryStats

time_start = time.clock()

dt = 1e-2
params, labels = obs_params(reduced_model=True)
# params = np.array([5, 0.4])  # for stability test

I, t, t_on, t_off = syn_current(duration=150, dt=dt)

# define models
dap = DAP(-75, params)
dap_exp = DAPExp(-75, params)
dap_feexp = DAPFeExp(-75, params)
dap_be = DAPBe(-75, params)

# run models
DAPdict = dap.simulate(dt, t, I, channels=True, noise=True, noise_fact=1e-1)
DAPexpDict = dap_exp.simulate(dt, t, I, channels=True, noise=True, noise_fact=1e-1)
DAPfexpDict = dap_feexp.simulate(dt, t, I, channels=True, noise=True, noise_fact=1e-1)
DAPbeDict = dap_be.simulate(dt, t, I, channels=True, noise=False, noise_fact=1e-1)

sum_stats_dict = DAPSummaryStatsDict(t_on, t_off, n_summary=8)
sum_stats = DAPSummaryStats(t_on, t_off, n_summary=8)
x_o =  {'data': DAPbeDict['U'],
        'time': t,
        'dt': dt,
        'I': I}
# print('summary stats:', sum_stats_dict.calc([x_o]))
print('summary stats A:', sum_stats.calc([x_o]))
# print('ss diff:', sum_stats_dict.calc([x_o]) - sum_stats.calc([x_o]), '\n')


time_end = time.clock()
print('time elapsed:', time_end - time_start)

# plot voltage trace
fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(20, 10));
ax[0][0].plot(t, DAPdict['U'], label='DAP');
ax[0][0].set_title('Forward Euler')
ax[0][0].grid()

ax[1][0].plot(t, DAPexpDict['U'], label='DAPExp');
ax[1][0].set_title('Exp Euler')
ax[1][0].grid()

ax[2][0].plot(t, DAPfexpDict['U'], label='DAPExp2');
ax[2][0].set_title('Exp + Forward Euler')
ax[2][0].grid()

ax[3][0].plot(t, DAPbeDict['U'], label='DapBe');
ax[3][0].set_title('Backward Euler')
ax[3][0].grid()

ax[4][0].plot(t, I);


# plot activation functions
ax[0][1].plot(t, DAPdict['M_nap'], label='M_nap');
ax[0][1].plot(t, DAPdict['M_nat'], label='M_nat');
ax[0][1].plot(t, DAPdict['H_nap'], label='H_nap');
ax[0][1].plot(t, DAPdict['H_nat'], label='H_nat');
ax[0][1].plot(t, DAPdict['N_hcn'], label='N_hcn');
ax[0][1].plot(t, DAPdict['N_kdr'], label='N_kdr');
ax[0][1].set_title('Forward Euler')
ax[0][1].legend()

ax[1][1].plot(t, DAPexpDict['M_nap'], label='M_nap_exp');
ax[1][1].plot(t, DAPexpDict['M_nat'], label='M_nat_exp');
ax[1][1].plot(t, DAPexpDict['H_nap'], label='H_nap_exp');
ax[1][1].plot(t, DAPexpDict['H_nat'], label='H_nat_exp');
ax[1][1].plot(t, DAPexpDict['N_hcn'], label='N_hcn_exp');
ax[1][1].plot(t, DAPexpDict['N_kdr'], label='N_kdr_exp');
ax[1][1].set_title('Exp Euler')
ax[1][1].legend()

ax[2][1].plot(t, DAPfexpDict['M_nap'], label='M_nap_exp2');
ax[2][1].plot(t, DAPfexpDict['M_nat'], label='M_nat_exp2');
ax[2][1].plot(t, DAPfexpDict['H_nap'], label='H_nap_exp2');
ax[2][1].plot(t, DAPfexpDict['H_nat'], label='H_nat_exp2');
ax[2][1].plot(t, DAPfexpDict['N_hcn'], label='N_hcn_exp2');
ax[2][1].plot(t, DAPfexpDict['N_kdr'], label='N_kdr_exp2');
ax[2][1].set_title('Exp + Forward Euler')
ax[2][1].legend()

ax[3][1].plot(t, DAPbeDict['M_nap'], label='M_napBe');
ax[3][1].plot(t, DAPbeDict['M_nat'], label='M_natBe');
ax[3][1].plot(t, DAPbeDict['H_nap'], label='H_napBe');
ax[3][1].plot(t, DAPbeDict['H_nat'], label='H_natBe');
ax[3][1].plot(t, DAPbeDict['N_hcn'], label='N_hcnBe');
ax[3][1].plot(t, DAPbeDict['N_kdr'], label='N_kdrBe');
ax[3][0].set_title('Backward Euler')
ax[3][1].legend()

plt.show()
