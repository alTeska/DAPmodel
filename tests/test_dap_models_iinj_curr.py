import time
import matplotlib.pyplot as plt
from dap import DAP, DAPBe, DAPExp, DAPFeExp
from dap.utils import load_current, obs_params


data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
# data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_11d.dat'  # second best cell

time_start = time.clock()

# load the data
I, v, t, t_on, t_off, dt = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)
params, labels = obs_params()

# define models
dap = DAP(-75, params)
dap_exp = DAPExp(-75, params)
dap_feexp = DAPFeExp(-75, params)
dap_be = DAPBe(-75, params)

# run model
DAPdict = dap.simulate(dt, t, I, channels=True)
DAPexpDict = dap_exp.simulate(dt, t, I, channels=True)
DAPfexpDict = dap_feexp.simulate(dt, t, I, channels=True)
DAPbeDict = dap_be.simulate(dt, t, I, channels=True)

time_end = time.clock()
print('time elapsed:', time_end - time_start)

# plot voltage trace
fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(20, 10))
ax[0][0].plot(t, DAPdict['U'], label='DAP')
ax[0][0].set_title('Forward Euler')
ax[0][0].grid()

ax[1][0].plot(t, DAPexpDict['U'], label='DAPExp')
ax[1][0].set_title('Exp Euler')
ax[1][0].grid()

ax[2][0].plot(t, DAPfexpDict['U'], label='DAPExp2')
ax[2][0].set_title('Exp + Forward Euler')
ax[2][0].grid()

ax[3][0].plot(t, DAPbeDict['U'], label='DapBe')
ax[3][0].set_title('Backward Euler')
ax[3][0].grid()

ax[4][0].plot(t, I)

# plot activation functions
ax[0][1].plot(t, DAPdict['M_nap'], label='M_nap')
ax[0][1].plot(t, DAPdict['M_nat'], label='M_nat')
ax[0][1].plot(t, DAPdict['H_nap'], label='H_nap')
ax[0][1].plot(t, DAPdict['H_nat'], label='H_nat')
ax[0][1].plot(t, DAPdict['N_hcn'], label='N_hcn')
ax[0][1].plot(t, DAPdict['N_kdr'], label='N_kdr')
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
