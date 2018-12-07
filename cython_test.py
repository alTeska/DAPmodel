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

dap_cython = DAPcython(-75, params)
dap_cython.simulate(dt, t, I)


# print(timeit.timeit(lambda: DAP_C.x_inf(1, 2, 3), number=int(1e5)))
# print(timeit.timeit(lambda: DAP.x_inf(1, 2, 3)  , number=int(1e5)))
