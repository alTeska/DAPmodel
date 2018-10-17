import numpy as np

class DAP_Model():
    """
    Hodgkin Huxley modified to DAP Model for the tests of LFI
    Model conists of 4 types of ion channels: Nap, Nat, Kdr, hcn_slow.
    """
    nois_fact_obs = 0.
    seed = None
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    cm = 0.63      #* 1e-6  # uF
    diam = 50.0    * 1e-4  # cm
    L = 100.0      * 1e-4  # cm
    cell_area = diam * L * np.pi
    Ra = 100.0

    e_hcn = -29.46    #* 1e-3  # mV
    e_nap = 60.0      #* 1e-3  # mV
    e_nat = 60.0      #* 1e-3  # mV
    e_leak = -86.53   #* 1e-3  # mV
    g_leak = 0.000430

    # kdr
    n_pow_kdr = 4


    n_vh_kdr = -68.29         # * 1e-3  # mV
    n_vs_kdr = 18.84          # * 1e-3  # mV
    n_tau_min_kdr = 0.286     # * 1e-3  # ms
    n_tau_max_kdr = 21.286    # * 1e-3  # ms
    n_tau_delta_kdr = 0.746   # * 1e-3  # ms


    e_kdr = -110.0    #* 1e-3  # mV
    gbar_kdr = 0.00313   # (S/cm2)

    kdr_n = {
        'pow' : 4,
        'vh' : -68.29,         # mV
        'vs' : 18.84,         # mV
        'tau_min' : 0.286,     # ms
        'tau_max' : 21.286,    # ms
        'tau_delta' : 0.746,   # ms
    }

    # hcn_slow
    n_pow_hcn = 1

    gbar_hcn = 5e-05   # (S/cm2)

    n_vh_hcn = -77.9        # * 1e-3  # mV
    n_vs_hcn = -20.54       # * 1e-3  # mV
    n_tau_min_hcn = 2.206   # * 1e-3  # ms
    n_tau_max_hcn = 137.799 # * 1e-3  # ms
    n_tau_delta_hcn = 0.21  # * 1e-3  # ms


    # Nat
    gbar_nat = 0.142   # (S/cm2)

    m_pow_nat = 3
    h_pow_nat = 1

    m_vs_nat = 11.99        #* 1e-3  # mV
    h_vs_nat = -13.17       #* 1e-3  # mV
    m_vh_nat = -30.94       #* 1e-3  # mV
    h_vh_nat = -60.44       #* 1e-3  # mV
    m_tau_min_nat = 0       #* 1e-3  # ms
    h_tau_min_nat = 0.001   #* 1e-3  # ms
    m_tau_max_nat = 0.193   #* 1e-3  # ms
    h_tau_max_nat = 8.743   #* 1e-3  # ms
    h_tau_delta_nat = 0.44  #* 1e-3  # ms
    m_tau_delta_nat = 0.187 #* 1e-3  # ms

    # Nap
    gbar_nap = 0.01527  # (S/cm2)

    m_pow_nap = 3  # p
    h_pow_nap = 1  # q

    m_vs_nap = 16.11    #* 1e-3  # mV
    h_vs_nap = -19.19   #* 1e-3  # mV
    m_vh_nap = -52.82   #* 1e-3  # mV
    h_vh_nap = -82.54   #* 1e-3  # mV
    m_tau_min_nap = 0.036   #* 1e-3  # ms
    h_tau_min_nap = 0.336   #* 1e-3  # ms
    m_tau_max_nap = 15.332  #* 1e-3  # ms
    h_tau_max_nap = 13.659  #* 1e-3  # ms
    m_tau_delta_nap = 0.505 # * 1e-3  # ms
    h_tau_delta_nap = 0.439 # * 1e-3  # ms

    def temp_corr(self, temp):
        '''temperature correction'''
        return 3**(0.1*(temp-6.3))

    def x_inf(self, V, x_vh, x_vs):
        '''steady state values'''
        return 1 / (1 + np.exp((x_vh - V) / x_vs))

    def dx_dt(self, x, x_inf, x_tau):
        return (x_inf - x) / x_tau

    def x_tau(self, V, x_tau_min, x_tau_max, xinf, x_tau_delta, x_vs, x_vh):
        return (x_tau_min + (x_tau_max - x_tau_min) * xinf * \
                np.exp(x_tau_delta * (x_vh - V) / x_vs))

    def x_tau_dict(self, V, xinf, ion_ch):
        return (ion_ch['tau_min'] + (ion_ch['tau_max'] - ion_ch['tau_min']) * \
                xinf * np.exp(ion_ch['tau_delta'] * \
                (ion_ch['vh'] - V) / ion_ch['vs']))


    def i_na(self, V, m, h, gbar, m_pow, h_pow, e_ion):
        return (gbar * self.cell_area) * m**m_pow * h**h_pow * (V - e_ion)

    def i_k(self, V, n, gbar, n_pow, e_ion):
        return (gbar * self.cell_area) * n**n_pow * (V - e_ion)

    def i_k_dict(self, V, n, gbar, n_pow, e_ion):
        return (gbar * self.cell_area) * n**n_pow * (V - e_ion)


    def simulate(self, T, dt, i_inj):

        '''run simulation of DAP model given the injection current'''
        t = np.linspace(0, T, int(T/dt))
        U = np.zeros_like(t)
        i_inj = i_inj * 1e-3  # input should be in uA (nA * 1e-3)

        M_nap = np.zeros_like(t)
        M_nat = np.zeros_like(t)
        H_nap = np.zeros_like(t)
        H_nat = np.zeros_like(t)
        N_hcn = np.zeros_like(t)
        N_kdr = np.zeros_like(t)

        U[0] = -75 #* 1e-3   # mV
        M_nap[0] = self.x_inf(U[0], self.m_vh_nap, self.m_vs_nap)
        M_nat[0] = self.x_inf(U[0], self.m_vh_nat, self.m_vs_nat)
        H_nap[0] = self.x_inf(U[0], self.h_vh_nap, self.h_vs_nap)
        H_nat[0] = self.x_inf(U[0], self.h_vh_nat, self.h_vs_nat)
        N_hcn[0] = self.x_inf(U[0], self.n_vh_hcn, self.n_vs_hcn)
        N_kdr[0] = self.x_inf(U[0], self.n_vh_kdr, self.n_vs_kdr)


        for n in range(0, len(i_inj)-1):
            # calculate steady states
            M_nap_inf = self.x_inf(U[n], self.m_vh_nap, self.m_vs_nap)
            M_nat_inf = self.x_inf(U[n], self.m_vh_nat, self.m_vs_nat)
            H_nap_inf = self.x_inf(U[n], self.h_vh_nap, self.h_vs_nap)
            H_nat_inf = self.x_inf(U[n], self.h_vh_nat, self.h_vs_nat)

            N_hcn_inf = self.x_inf(U[n], self.n_vh_hcn, self.n_vs_hcn)
            N_kdr_inf = self.x_inf(U[n], self.n_vh_kdr, self.n_vs_kdr)


            # calcualte  time constants
            tau_m_nap = self.x_tau(U[n], self.m_tau_min_nap, self.m_tau_max_nap,
                                   M_nap_inf, self.m_tau_delta_nap, self.m_vs_nap,
                                   self.m_vh_nap)
            tau_m_nat = self.x_tau(U[n], self.m_tau_min_nat, self.m_tau_max_nat,
                                   M_nat_inf, self.m_tau_delta_nat, self.m_vs_nat,
                                   self.m_vh_nat)

            tau_h_nap = self.x_tau(U[n], self.h_tau_min_nap, self.h_tau_max_nap,
                                   H_nap_inf, self.h_tau_delta_nap,
                                   self.h_vs_nap, self.h_vh_nap)
            tau_h_nat = self.x_tau(U[n], self.h_tau_min_nat, self.h_tau_max_nat,
                                   H_nat_inf, self.h_tau_delta_nat,
                                   self.h_vs_nat, self.h_vh_nat)

            tau_n_hcn = self.x_tau(U[n], self.n_tau_min_hcn, self.n_tau_max_hcn,
                                   N_hcn_inf, self.n_tau_delta_hcn,
                                   self.n_vs_hcn, self.n_vh_hcn)
            # tau_n_kdr = self.x_tau(U[n], self.n_tau_min_kdr, self.n_tau_max_kdr,
            #                        N_kdr_inf, self.n_tau_delta_kdr,
            #                        self.n_vs_kdr, self.n_vh_kdr)

            tau_n_kdr = self.x_tau_dict(U[n], N_kdr_inf, self.kdr_n)


            # calculate all steady states
            M_nap[n+1] = M_nap[n] + self.dx_dt(M_nap[n], M_nap_inf, tau_m_nap) * dt
            M_nat[n+1] = M_nat[n] + self.dx_dt(M_nat[n], M_nat_inf, tau_m_nat) * dt
            H_nap[n+1] = H_nap[n] + self.dx_dt(H_nap[n], H_nap_inf, tau_h_nap) * dt
            H_nat[n+1] = H_nat[n] + self.dx_dt(H_nat[n], H_nat_inf, tau_h_nat) * dt
            N_hcn[n+1] = N_hcn[n] + self.dx_dt(N_hcn[n], N_hcn_inf, tau_n_hcn) * dt
            N_kdr[n+1] = N_kdr[n] + self.dx_dt(N_kdr[n], N_kdr_inf, tau_n_kdr) * dt


            # calculate ionic currents
            i_nap = self.i_na(U[n], M_nap[n+1], H_nap[n+1], self.gbar_nap,
                              self.m_pow_nap, self.h_pow_nap, self.e_nap)

            i_nat = self.i_na(U[n], M_nat[n+1], H_nat[n+1], self.gbar_nat,
                              self.m_pow_nat, self.h_pow_nat, self.e_nat)

            i_hcn = self.i_k(U[n], N_hcn[n+1], self.gbar_hcn, self.n_pow_hcn,
                             self.e_hcn)
            i_kdr = self.i_k(U[n], N_kdr[n+1], self.gbar_kdr, self.n_pow_kdr,
                             self.e_kdr)


            i_ion = (i_nap + i_nat + i_kdr + i_hcn) * 1e3

            # calculate membrane potential
            i_leak = (self.g_leak * self.cell_area) * (U[n] - self.e_leak) * 1e3
            U[n+1] = U[n] + (-i_ion - i_leak + i_inj[n])/(self.cm*self.cell_area) * dt

        return U #+ self.nois_fact_obs*self.rng.randn(t.shape[0],1)
        # return U, M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr
