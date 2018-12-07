import timeit
import numpy as np
# import matplotlib.pyplot as plt

from DAPmodel import DAP
from DAPmodel.DAPcython import DAPcython
from DAPmodel.DAPbaseC import DAPCython
from DAPmodel.utils import obs_params, syn_current


dt = 1e-2
params, labels = obs_params()

I, t, t_on, t_off = syn_current(duration=10, dt=dt)

# define models
dap = DAP(-75, params)
dap_cython = DAPcython(-75, params)

# run models
dap.simulate(dt, t, I)
dap_cython.simulate(dt, t, I)

print(timeit.timeit(lambda: dap_cython.simulate(dt, t, I), number=int(1e2)))
print(timeit.timeit(lambda: dap.simulate(dt, t, I)       , number=int(1e2)))
