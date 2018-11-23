import time
import numpy as np
import matplotlib.pyplot as plt
from DAPmodel import DAP, DAPBe, DAPExp, DAPFeExp
from DAPmodel import obs_params, syn_current


time_start = time.clock()

dt = 1e-2
params, labels = obs_params()
#   params = np.array([16, 0.4])  # for stability test

I, t, t_on, t_off = syn_current(duration=150, dt=dt)

# define model
dap = DAP(-75, params)
dap_exp = DAPExp(-75, params)
dap_feexp = DAPFeExp(-75, params)
dap_be = DAPBe(-75, params)

# run model
UDap, M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr = dap.simulate(dt, t, I)
UDapExp, M_nap_exp, M_nat_exp, H_nap_exp, H_nat_exp, N_hcn_exp, N_kdr_exp = dap_exp.simulate(dt, t, I)
UDapFeExp, M_nap_feexp, M_nat_feexp, H_nap_feexp, H_nat_feexp, N_hcn_feexp, N_kdr_feexp = dap_feexp.simulate(dt, t, I)
UDapBe, M_napBe, M_natBe, H_napBe, H_natBe, N_hcnBe, N_kdrBe = dap_be.simulate(dt, t, I)

time_end = time.clock()
print('time elapsed:', time_end - time_start)

# plot voltage trace
fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(20, 10));
ax[0][0].plot(t, UDap, label='DAP');
ax[0][0].set_title('Forward Euler')
ax[0][0].grid()

ax[1][0].plot(t, UDapExp, label='DAPExp');
ax[1][0].set_title('Exp Euler')
ax[1][0].grid()

ax[2][0].plot(t, UDapFeExp, label='DAPExp2');
ax[2][0].set_title('Exp + Forward Euler')
ax[2][0].grid()

ax[3][0].plot(t, UDapBe, label='DapBe');
ax[3][0].set_title('Backward Euler')
ax[3][0].grid()

ax[4][0].plot(t, I);


# plot activation functions
ax[0][1].plot(t, M_nap, label='M_nap');
ax[0][1].plot(t, M_nat, label='M_nat');
ax[0][1].plot(t, H_nap, label='H_nap');
ax[0][1].plot(t, H_nat, label='H_nat');
ax[0][1].plot(t, N_hcn, label='N_hcn');
ax[0][1].plot(t, N_kdr, label='N_kdr');
ax[0][1].set_title('Forward Euler')
ax[0][1].legend()

ax[1][1].plot(t, M_nap_exp, label='M_nap_exp');
ax[1][1].plot(t, M_nat_exp, label='M_nat_exp');
ax[1][1].plot(t, H_nap_exp, label='H_nap_exp');
ax[1][1].plot(t, H_nat_exp, label='H_nat_exp');
ax[1][1].plot(t, N_hcn_exp, label='N_hcn_exp');
ax[1][1].plot(t, N_kdr_exp, label='N_kdr_exp');
ax[1][1].set_title('Exp Euler')
ax[1][1].legend()

ax[2][1].plot(t, M_nap_feexp, label='M_nap_exp2');
ax[2][1].plot(t, M_nat_feexp, label='M_nat_exp2');
ax[2][1].plot(t, H_nap_feexp, label='H_nap_exp2');
ax[2][1].plot(t, H_nat_feexp, label='H_nat_exp2');
ax[2][1].plot(t, N_hcn_feexp, label='N_hcn_exp2');
ax[2][1].plot(t, N_kdr_feexp, label='N_kdr_exp2');
ax[2][1].set_title('Exp + Forward Euler')
ax[2][1].legend()

ax[3][1].plot(t, M_napBe, label='M_napBe');
ax[3][1].plot(t, M_natBe, label='M_natBe');
ax[3][1].plot(t, H_napBe, label='H_napBe');
ax[3][1].plot(t, H_natBe, label='H_natBe');
ax[3][1].plot(t, N_hcnBe, label='N_hcnBe');
ax[3][1].plot(t, N_kdrBe, label='N_kdrBe');
ax[3][0].set_title('Backward Euler')
ax[3][1].legend()

plt.show()
