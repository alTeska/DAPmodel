import numpy as np
from . import dap_cython
from . import dap_cython_be

# solver = dap_cython.forwardeuler
# solver = dap_cython_be.backwardeuler

# def solver(t, I, U, M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr, dt, rmat):
# def solver(t, I, V, m, n, h, p, q, r, u, dt, r_mat)
# t: array of time steps
# I: array of I values
# V: array of V values (OUTPUT)
# M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr: buffers for gating variables
# dt: time step
# r_mat: array of random inputs (voltage noise)
#
# The arrays must have the same size. The simulation runs until V is exhausted.


class DAPcython(object):
    def __init__(self, state, params, seed=None, solver=2, **kwargs):
        self.state = np.asarray(state)
        self.params = np.asarray(params)


        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

        solver = solver
        if solver == 1:
            self.solver = dap_cython.forwardeuler
            self.setparams = dap_cython.setparams
            self.setnoisefactor = dap_cython.setnoisefactor
        else:
            self.solver = dap_cython_be.backwardeuler
            self.setparams = dap_cython_be.setparams
            self.setnoisefactor = dap_cython_be.setnoisefactor


    def simulate(self, dt, t, i_inj, channels=False, noise=False, noise_fact=1e-3):
        """Run simulation of DAP model given the injection current - CYTHON"""
        self.setparams(self.params)
        self.setnoisefactor(noise_fact)

        i_inj = i_inj * 1e-3  # input should be in uA (nA * 1e-3)

        U = np.zeros_like(t).astype(np.float64)
        M_nap = np.zeros_like(t).astype(np.float64)
        M_nat = np.zeros_like(t).astype(np.float64)
        H_nap = np.zeros_like(t).astype(np.float64)
        H_nat = np.zeros_like(t).astype(np.float64)
        N_hcn = np.zeros_like(t).astype(np.float64)
        N_kdr = np.zeros_like(t).astype(np.float64)
        r_mat = self.rng.randn(len(t)).astype(np.float64)

        U[0] = self.state


        self.solver(t, i_inj, U, M_nap, M_nat, H_nap, H_nat, N_hcn,
                                N_kdr, dt, r_mat)

        if noise:
            U = U + noise_fact*self.rng.randn(1, t.shape[0])

        if channels:
            return {
                    'U': U.reshape(-1,1),
                    'M_nap': M_nap,
                    'M_nat': M_nat,
                    'H_nap': H_nap,
                    'H_nat': H_nat,
                    'N_hcn': N_hcn,
                    'N_kdr': N_kdr,
                    }
        else:
            return U.reshape(-1,1)
