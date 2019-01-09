import numpy as np
import matplotlib.pyplot as plt

from DAPmodel import DAP, DAPBe
from DAPmodel import DAPcython
from DAPmodel.utils import obs_params, syn_current


# ranges to be tested:
# gbar_nap       [0   ; 0.5]   ( 0.01527)
# nap_m_vs       [1   ; 30 ]   ( 16.11  )
# nap_m_tau_max  [0   ; 100]   ( 15.332 )
# nap_h_vs       [-30 ;-1  ]   (-19.19  )
# nap_h_tau_max  [0   ; 100]   ( 13.659 )
# nat_m_vh       [-100; 0  ]   (-30.94  )
# nat_h_vh       [-100; 0  ]   (-60.44  )
# nat_m_vs       [ 1  ; 30 ]   ( 11.99  )
# nat_h_vs       [-30 ;-1  ]   (-13.17  )
# kdr_n_vs       [ 1  ; 30 ]   ( 18.84  )

gbar_range = np.arange(0, 0.5, 0.1)
print(gbar_range)

dt = 1e-2
params, labels = obs_params()
I, t, t_on, t_off = syn_current(duration=120, dt=dt)

for r in gbar_range:
    params[0] = r
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

    fig, ax = plt.subplots(2, 1, figsize=(20, 10));
    ax[0].plot(t, U, label='forward')
    ax[0].plot(t, U_cython, label='forward_cython')
    ax[0].legend()
    ax[1].plot(t, U_back, label='backward')
    ax[1].plot(t, U_cython_back, label='backward_cython')
    ax[1].legend()

    plt.show()
