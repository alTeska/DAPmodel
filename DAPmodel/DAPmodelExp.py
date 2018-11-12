import numpy as np

class DAPExp():
    """
    DAP Model based on HH equations for the tests with LFI
    Model conists of 4 types of ion channels: Nap, Nat, Kdr, hcn_slow.
    i_inj = nA
    """

    def __init__(self, init, params, seed=None):
        self.state = np.asarray(init)
        self.params = np.asarray(params)

        # Nap
        self.nap_m = {
            'pow': 3,
            'vs': 16.11,      # mV # 16.11 params[1]
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
            'tau_max': 13.659,  # ms  # 13.659 params[2]
            'tau_delta': params[1],    # ms
            }


        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()


    gbar_kdr = 0.00313  # (S/cm2)
    gbar_hcn = 5e-05    # (S/cm2)
    gbar_nap = 0.01527  # (S/cm2)
    gbar_nat = 0.142    # (S/cm2)

    cm = 0.63      #* 1e-6  # uF/cm2
    diam = 50.0    * 1e-4  # cm
    L = 100.0      * 1e-4  # cm
    Ra = 100.0
    cell_area = diam * L * np.pi

    gbar_kdr = gbar_kdr * cell_area  # S
    gbar_hcn = gbar_hcn * cell_area  # S
    gbar_nap = gbar_nap * cell_area  # S
    gbar_nat = gbar_nat * cell_area  # S

    cm = cm * cell_area  # uF


    e_hcn = -29.46    # mV
    e_nap = 60.0      # mV
    e_nat = 60.0      # mV
    e_leak = -86.53   # mV
    e_kdr = -110.0    # mV
    g_leak = 0.000430
    g_leak = g_leak * cell_area


    kdr_n = {
        'pow': 4,
        'vh': -68.29,         # mV
        'vs': 18.84,          # mV
        'tau_min': 0.286,     # ms
        'tau_max': 21.286,    # ms
        'tau_delta': 0.746,   # ms
    }

    hcn_n = {
        'pow': 1,
        'vh': -77.9,         # mV
        'vs': -20.54,        # mV
        'tau_min': 2.206,    # ms
        'tau_max': 137.799,  # ms
        'tau_delta': 0.21,   # ms
    }

    # Nat

    nat_m = {
        'pow': 3,
        'vs': 11.99,         # mV
        'vh': -30.94,        # mV
        'tau_min': 0,        # ms
        'tau_max': 0.193,    # ms
        'tau_delta': 0.187,  # ms
    }

    nat_h = {
        'pow': 1,
        'vs': -13.17,       # mV
        'vh': -60.44,       # mV
        'tau_min': 0.001,   # ms
        'tau_max': 8.743,   # ms
        'tau_delta': 0.44,  # ms
    }


    def temp_corr(self, temp):
        '''temperature correction'''
        return 3**(0.1*(temp-6.3))

    def x_inf(self, V, x_vh, x_vs):
        '''steady state values'''
        return 1 / (1 + np.exp((x_vh - V) / x_vs))

    def dx_dt(self, x, x_inf, x_tau, dt):
        '''differential equations for m,h,n'''
        return (x_inf - x) / x_tau


    def dx_plus(self, x, x_inf, x_tau, dt):
        '''differential equations for m,h,n'''
        return x_inf + (x - x_inf) * np.exp(-dt/x_tau)


    def x_tau(self, V, xinf, ion_ch):
        return (ion_ch['tau_min'] + (ion_ch['tau_max'] - ion_ch['tau_min']) * \
                xinf * np.exp(ion_ch['tau_delta'] * \
                (ion_ch['vh'] - V) / ion_ch['vs']))


    def i_na(self, V, m, h, gbar, m_pow, h_pow, e_ion):
        '''calculates sodium-like ion current'''
        return gbar * m**m_pow * h**h_pow * (V - e_ion)

    def i_k(self, V, n, gbar, n_pow, e_ion):
        '''calculates potasium-like ion current'''
        return gbar * n**n_pow * (V - e_ion)



    def g_e_na(self, m, h, gbar, m_pow, h_pow, e_ion):
        '''calculates sodium-like ion current'''
        return (gbar) * m**m_pow * h**h_pow * (e_ion)

    def g_e_k(self, n, gbar, n_pow, e_ion):
        '''calculates potasium-like ion current'''
        return (gbar) * n**n_pow * (e_ion)

    def g_na(self, m, h, gbar, m_pow, h_pow):
        '''calculates sodium-like ion current'''
        return (gbar) * m**m_pow * h**h_pow

    def g_k(self, n, gbar, n_pow):
        '''calculates potasium-like ion current'''
        return (gbar) * n**n_pow


    def simulate(self, dt, t, i_inj):
        """run simulation of DAP model given the injection current

        Parameters
        ----------
        dt : float
            Timestep
        t : float
            Numpy array with time course
        i_inj : array
            Numpy array with the input I
        """
        nois_fact_obs = 0.00001


        U = np.zeros_like(t)
        i_inj = i_inj * 1e-3  # input should be in uA (nA * 1e-3)

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

        for n in range(0, len(i_inj)-1):
            # calculate ionic currents
            g_e_nap = self.g_e_na(M_nap[n], H_nap[n], self.gbar_nap,
                              self.nap_m['pow'], self.nap_h['pow'], self.e_nap)
            g_e_nat = self.g_e_na(M_nat[n], H_nat[n], self.gbar_nat,
                              self.nat_m['pow'], self.nat_h['pow'], self.e_nat)
            g_e_hcn = self.g_e_k(N_hcn[n], self.gbar_hcn, self.hcn_n['pow'], self.e_hcn)
            g_e_kdr = self.g_e_k(N_kdr[n], self.gbar_kdr, self.kdr_n['pow'], self.e_kdr)

            g_e_leak = (self.g_leak) * (self.e_leak)
            g_e_sum = g_e_leak + (g_e_nap + g_e_nat + g_e_kdr + g_e_hcn)

            g_nap = self.g_na(M_nap[n], H_nap[n], self.gbar_nap, self.nap_m['pow'], self.nap_h['pow'])
            g_nat = self.g_na(M_nat[n], H_nat[n], self.gbar_nat, self.nat_m['pow'], self.nat_h['pow'])
            g_hcn = self.g_k(N_hcn[n], self.gbar_hcn, self.hcn_n['pow'])
            g_kdr = self.g_k(N_kdr[n], self.gbar_kdr, self.kdr_n['pow'])

            g_sum = (g_nap + g_nat + g_hcn + g_kdr + self.g_leak)


            # calculate membrane potential
            V_inf = (g_e_sum + i_inj[n]*1e3) / g_sum
            tau_v = (self.cm) * 1e3 / g_sum
            # print(U[n])
            U[n+1] = V_inf + (U[n] - V_inf) * np.exp(-dt / tau_v)

            # U[n+1] = U[n] + (-i_ion - i_leak + i_inj[n])/(self.cm) * dt

            # calculate x_inf
            M_nap_inf = self.x_inf(U[n+1], self.nap_m['vh'], self.nap_m['vs'])
            M_nat_inf = self.x_inf(U[n+1], self.nat_m['vh'], self.nat_m['vs'])
            H_nap_inf = self.x_inf(U[n+1], self.nap_h['vh'], self.nap_h['vs'])
            H_nat_inf = self.x_inf(U[n+1], self.nat_h['vh'], self.nat_h['vs'])
            N_hcn_inf = self.x_inf(U[n+1], self.hcn_n['vh'], self.hcn_n['vs'])
            N_kdr_inf = self.x_inf(U[n+1], self.kdr_n['vh'], self.kdr_n['vs'])

            # calcualte  time constants
            tau_m_nap = self.x_tau(U[n+1], M_nap_inf, self.nap_m)
            tau_h_nap = self.x_tau(U[n+1], H_nap_inf, self.nap_h)
            tau_m_nat = self.x_tau(U[n+1], M_nat_inf, self.nat_m)
            tau_h_nat = self.x_tau(U[n+1], H_nat_inf, self.nat_h)
            tau_n_hcn = self.x_tau(U[n+1], N_hcn_inf, self.hcn_n)
            tau_n_kdr = self.x_tau(U[n+1], N_kdr_inf, self.kdr_n)

            # calculate all steady states
            M_nap[n+1] = self.dx_plus(M_nap[n], M_nap_inf, tau_m_nap, dt)
            M_nat[n+1] = self.dx_plus(M_nat[n], M_nat_inf, tau_m_nat, dt)
            H_nap[n+1] = self.dx_plus(H_nap[n], H_nap_inf, tau_h_nap, dt)
            H_nat[n+1] = self.dx_plus(H_nat[n], H_nat_inf, tau_h_nat, dt)
            N_hcn[n+1] = self.dx_plus(N_hcn[n], N_hcn_inf, tau_n_hcn, dt)
            N_kdr[n+1] = self.dx_plus(N_kdr[n], N_kdr_inf, tau_n_kdr, dt)

        # return U.reshape(-1,1) #+ nois_fact_obs*self.rng.randn(t.shape[0],1)
        return U.reshape(-1,1), M_nap, M_nat, H_nap