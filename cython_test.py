import timeit
import numpy as np

from DAPmodel.DAPbase import DAPBase
from DAPmodel.DAPbaseC import DAPBaseC

states = -75
params = np.array([5, 0.4])


DAP_C = DAPBaseC(states, params)
DAP = DAPBase(states, params)


DAP_C.x_inf(1, 2, 3)
DAP.x_inf(1, 2, 3)

def run_DAP_C():
    DAP_C.x_inf(1, 2, 3)
    pass

def run_DAP():
    DAP.x_inf(1, 2, 3)
    pass


print(timeit.timeit(lambda: run_DAP_C(), number=1))
print(timeit.timeit(lambda: run_DAP(), number=1))
