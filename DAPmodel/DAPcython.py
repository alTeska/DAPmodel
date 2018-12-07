import numpy as np

from . import dap_cython

# solver = dap_cython.forwardeuler

# def solver(t, I, V, m, n, h, p, q, r, u, dt, r_mat)
# t: array of time steps
# I: array of I values
# V: array of V values (OUTPUT)
# m, n, h, p, q, r, u: buffers for gating variables
# dt: time step
# r_mat: array of random inputs (voltage noise)
#
# The arrays must have the same size. The simulation runs until V is exhausted.

class DAPcython(object):
    def __init__(self, state, params, seed=None):
        self.state = np.asarray(state)
        self.params = np.asarray(params)

        def __init__(self, state, params, seed=None, **kwargs):
            super(DAPBase, self).__init__(**kwargs)
            self.state = np.asarray(state)
            self.params = np.asarray(params)

            # Nap
            self.nap_m = {
                'pow': 3,
                'vs': 16.11,          # mV
                'vh': -52.82,         # mV
                'tau_min': 0.036,     # ms
                'tau_max': params[0], # ms # 15.332
                'tau_delta': 0.505,   # ms
                }

            self.nap_h = {
                'pow': 1,
                'vs': -19.19,          # mV
                'vh': -82.54,          # mV
                'tau_min': 0.336,      # ms
                'tau_max': 13.659,     # ms  # 13.659 param s[2]
                'tau_delta': params[1],# ms
                }
            self.kdr_n = {
                'pow': 4,
                'vh': -68.29,         # mV
                'vs': 18.84,          # mV
                'tau_min': 0.286,     # ms
                'tau_max': 21.286,    # ms
                'tau_delta': 0.746,   # ms
            }

            self.hcn_n = {
                'pow': 1,
                'vh': -77.9,         # mV
                'vs': -20.54,        # mV
                'tau_min': 2.206,    # ms
                'tau_max': 137.799,  # ms
                'tau_delta': 0.21,   # ms
            }

            self.nat_m = {
                'pow': 3,
                'vs': 11.99,         # mV
                'vh': -30.94,        # mV
                'tau_min': 0,        # ms
                'tau_max': 0.193,    # ms
                'tau_delta': 0.187,  # ms
            }

            self.nat_h = {
                'pow': 1,
                'vs': -13.17,       # mV
                'vh': -60.44,       # mV
                'tau_min': 0.001,   # ms
                'tau_max': 8.743,   # ms
                'tau_delta': 0.44,  # ms
            }


            self.gbar_kdr = 0.00313  # (S/cm2)
            self.gbar_hcn = 5e-05    # (S/cm2)
            self.gbar_nap = 0.01527  # (S/cm2)
            self.gbar_nat = 0.142    # (S/cm2)

            self.cm = 0.63      #* 1e-6  # uF/cm2
            self.diam = 50.0    * 1e-4  # cm
            self.L = 100.0      * 1e-4  # cm
            self.Ra = 100.0
            self.cell_area = self.diam * self.L * np.pi

            self.gbar_kdr = self.gbar_kdr * self.cell_area  # S
            self.gbar_hcn = self.gbar_hcn * self.cell_area  # S
            self.gbar_nap = self.gbar_nap * self.cell_area  # S
            self.gbar_nat = self.gbar_nat * self.cell_area  # S

            self.cm = self.cm * self.cell_area  # uF

            self.e_hcn = -29.46    # mV
            self.e_nap = 60.0      # mV
            self.e_nat = 60.0      # mV
            self.e_leak = -86.53   # mV
            self.e_kdr = -110.0    # mV
            self.g_leak = 0.000430
            self.g_leak = self.g_leak * self.cell_area


            self.seed = seed
            if seed is not None:
                self.rng = np.random.RandomState(seed=seed)
            else:
                self.rng = np.random.RandomState()


    def simulate(self, dt, t, i_inj, channels=False, noise=False, noise_fact=1e-3):
        """Run simulation of DAP model given the injection current - CYTHON"""

        i_inj = i_inj * 1e-3  # input should be in uA (nA * 1e-3)

        U = np.zeros_like(t).astype(np.float64)
        M_nap = np.zeros_like(t).astype(np.float64)
        M_nat = np.zeros_like(t).astype(np.float64)
        H_nap = np.zeros_like(t).astype(np.float64)
        H_nat = np.zeros_like(t).astype(np.float64)
        N_hcn = np.zeros_like(t).astype(np.float64)
        N_kdr = np.zeros_like(t).astype(np.float64)

        U[0] = self.state
        # solver(t, i_inj, U, M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr)


        i = 12
        print(i)
        cur, i = dap_cython.update_inf(i)
        print(i)
        print('cur in python', cur)

        # return np.array(U).reshape(-1,1)
