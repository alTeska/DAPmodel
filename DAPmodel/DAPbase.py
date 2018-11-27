import abc
import numpy as np


# class DAPBase(metaclass=abc.ABCMeta):
class DAPBase(object):
    __metaclass__ = abc.ABCMeta
    """Abstract Base class of DAP cell model

    DAP Model based on HH equations. Model conists of 4 types of ion channels:
    Nap, Nat, Kdr, hcn_slow.
    i_inj = nA

    Base class initiates all of the parameters of the DAP model and provides access to parametrs setu.abc
    """

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

    def temp_corr(self, temp):
        '''temperature correction'''
        return 3**(0.1*(temp-6.3))

    # model integration
    def x_inf(self, V, x_vh, x_vs):
        '''steady state values'''
        return 1 / (1 + np.exp((x_vh - V) / x_vs))

    def x_tau(self, V, xinf, ion_ch):
        return (ion_ch['tau_min'] + (ion_ch['tau_max'] - ion_ch['tau_min']) * \
                xinf * np.exp(ion_ch['tau_delta'] * \
                (ion_ch['vh'] - V) / ion_ch['vs']))

    # currents
    def i_na(self, V, m, h, gbar, m_pow, h_pow, e_ion):
        '''calculates sodium-like ion current'''
        return gbar * m**m_pow * h**h_pow * (V - e_ion)

    def i_k(self, V, n, gbar, n_pow, e_ion):
        '''calculates potasium-like ion current'''
        return gbar * n**n_pow * (V - e_ion)


    @abc.abstractmethod
    def simulate(self, dt, t, i_inj, channels=False):
        """Run simulation of DAP model given the injection current

        Parameters
        ----------
        dt (float): Timestep
        t  (array): array with time course
        i_inj (array): array with the input I

        Returns:
        U (array): array with voltage trace

        if channels=True: dictionary with arrays contatining voltage trace and activation gates:
            'U': U.reshape(-1,1),
            'M_nap': M_nap,
            'M_nat': M_nat,
            'H_nap': H_nap,
            'H_nat': H_nat,
            'N_hcn': N_hcn,
            'N_kdr': N_kdr,
        """

        nois_fact_obs = 1e-5
        i_inj = i_inj * 1e-3  # input should be in uA (nA * 1e-3)

        U = np.zeros_like(t)
        M_nap = np.zeros_like(t)
        M_nat = np.zeros_like(t)
        H_nap = np.zeros_like(t)
        H_nat = np.zeros_like(t)
        N_hcn = np.zeros_like(t)
        N_kdr = np.zeros_like(t)

        U[0] = self.state
        M_nap[0] = self.x_inf(U[0], self.nap_m['vh'], self.nap_m['vs'])
        M_nat[0] = self.x_inf(U[0], self.nat_m['vh'], self.nat_m['vs'])
        H_nap[0] = self.x_inf(U[0], self.nap_h['vh'], self.nap_h['vs'])
        H_nat[0] = self.x_inf(U[0], self.nat_h['vh'], self.nat_h['vs'])
        N_hcn[0] = self.x_inf(U[0], self.hcn_n['vh'], self.hcn_n['vs'])
        N_kdr[0] = self.x_inf(U[0], self.kdr_n['vh'], self.kdr_n['vs'])

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
            return U.reshape(-1,1) #+ nois_fact_obs*self.rng.randn(t.shape[0],1)
