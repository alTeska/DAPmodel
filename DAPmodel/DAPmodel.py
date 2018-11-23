import numpy as np
from .DAPbase import DAPBase

class DAP(DAPBase):
    """
    DAP Model based on HH equations for the tests with LFI
    Model conists of 4 types of ion channels: Nap, Nat, Kdr, hcn_slow.
    i_inj = nA

    Integrated with Forward(explicit) Euler method.

    """

    def __init__(self, state, params, seed=None, **kwargs):
        super().__init__(state=state, params=params,
                         seed=seed, **kwargs)



    def temp_corr(self, temp):
        '''temperature correction'''
        return 3**(0.1*(temp-6.3))

    def x_inf(self, V, x_vh, x_vs):
        '''steady state values'''
        return 1 / (1 + np.exp((x_vh - V) / x_vs))

    def dx_dt(self, x, x_inf, x_tau):
        '''differential equations for m,h,n'''
        return (x_inf - x) / x_tau

    def x_tau(self, V, xinf, ion_ch):
        return (ion_ch['tau_min'] + (ion_ch['tau_max'] - ion_ch['tau_min']) * \
                xinf * np.exp(ion_ch['tau_delta'] * \
                (ion_ch['vh'] - V) / ion_ch['vs']))


    def i_na(self, V, m, h, gbar, m_pow, h_pow, e_ion):
        '''calculates sodium-like ion current'''
        return (gbar ) * m**m_pow * h**h_pow * (V - e_ion)

    def i_k(self, V, n, gbar, n_pow, e_ion):
        '''calculates potasium-like ion current'''
        return (gbar ) * n**n_pow * (V - e_ion)



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
            # calculate x_inf
            M_nap_inf = self.x_inf(U[n], self.nap_m['vh'], self.nap_m['vs'])
            M_nat_inf = self.x_inf(U[n], self.nat_m['vh'], self.nat_m['vs'])
            H_nap_inf = self.x_inf(U[n], self.nap_h['vh'], self.nap_h['vs'])
            H_nat_inf = self.x_inf(U[n], self.nat_h['vh'], self.nat_h['vs'])
            N_hcn_inf = self.x_inf(U[n], self.hcn_n['vh'], self.hcn_n['vs'])
            N_kdr_inf = self.x_inf(U[n], self.kdr_n['vh'], self.kdr_n['vs'])

            # calcualte  time constants
            tau_m_nap = self.x_tau(U[n], M_nap_inf, self.nap_m)
            tau_h_nap = self.x_tau(U[n], H_nap_inf, self.nap_h)
            tau_m_nat = self.x_tau(U[n], M_nat_inf, self.nat_m)
            tau_h_nat = self.x_tau(U[n], H_nat_inf, self.nat_h)
            tau_n_hcn = self.x_tau(U[n], N_hcn_inf, self.hcn_n)
            tau_n_kdr = self.x_tau(U[n], N_kdr_inf, self.kdr_n)

            # calculate all steady states
            M_nap[n+1] = M_nap[n] + self.dx_dt(M_nap[n], M_nap_inf, tau_m_nap) * dt
            M_nat[n+1] = M_nat[n] + self.dx_dt(M_nat[n], M_nat_inf, tau_m_nat) * dt
            H_nap[n+1] = H_nap[n] + self.dx_dt(H_nap[n], H_nap_inf, tau_h_nap) * dt
            H_nat[n+1] = H_nat[n] + self.dx_dt(H_nat[n], H_nat_inf, tau_h_nat) * dt
            N_hcn[n+1] = N_hcn[n] + self.dx_dt(N_hcn[n], N_hcn_inf, tau_n_hcn) * dt
            N_kdr[n+1] = N_kdr[n] + self.dx_dt(N_kdr[n], N_kdr_inf, tau_n_kdr) * dt


            # calculate ionic currents
            i_nap = self.i_na(U[n], M_nap[n+1], H_nap[n+1], self.gbar_nap,
                              self.nap_m['pow'], self.nap_h['pow'], self.e_nap)
            i_nat = self.i_na(U[n], M_nat[n+1], H_nat[n+1], self.gbar_nat,
                              self.nat_m['pow'], self.nat_h['pow'], self.e_nat)
            i_hcn = self.i_k(U[n], N_hcn[n+1], self.gbar_hcn, self.hcn_n['pow'],
                             self.e_hcn)
            i_kdr = self.i_k(U[n], N_kdr[n+1], self.gbar_kdr, self.kdr_n['pow'],
                             self.e_kdr)

            i_ion = (i_nap + i_nat + i_kdr + i_hcn) * 1e3
            i_leak = (self.g_leak) * (U[n] - self.e_leak) * 1e3

            # calculate membrane potential
            U[n+1] = U[n] + (-i_ion - i_leak + i_inj[n])/(self.cm) * dt


        # return U.reshape(-1,1) #+ nois_fact_obs*self.rng.randn(t.shape[0],1)
        return U.reshape(-1,1), M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr
