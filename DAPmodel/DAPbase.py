import abc
import numpy as np

# class DAPBase(metaclass=abc.ABCMeta):
class DAPBase(object):
    __metaclass__ = abc.ABCMeta
    """Abstract Base class of DAP cell model

    DAP Model based on HH equations. Model conists of 4 types of ion channels:
    Nap, Nat, Kdr, hcn_slow.
    i_inj = nA
    """

    def __init__(self, state, params, seed=None, **kwargs):
        super(DAPBase, self).__init__(**kwargs)
        self.state = np.asarray(state)
        self.params = np.asarray(params)

        # Nap
        self.nap_m = {
            'pow': 3,
            'vs': 16.11,          # mV # 16.11 params[1]
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

        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    @abc.abstractmethod
    def simulate(self, dt, t, i_inj):
        pass
