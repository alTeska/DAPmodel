import numpy as np
from .DAPbase import DAPBase


class DAPFeExp(DAPBase):
    """
    DAP Model based on HH equations for the tests with LFI
    Model conists of 4 types of ion channels: Nap, Nat, Kdr, hcn_slow.
    i_inj = nA

    Integrated with mixture of Exponential and Forward Euler method.
    """

    def __init__(self, state, params, seed=None, **kwargs):
        super().__init__(state=state, params=params,
                         seed=seed, **kwargs)


    def dx_dt_exp(self, x, x_inf, x_tau, dt):
        '''differential equations for m,h,n'''
        return x_inf + (x - x_inf) * np.exp(-dt/x_tau)

    # condactivities and conductances
    def g_e_na(self, m, h, gbar, m_pow, h_pow, e_ion):
        '''calculates sodium-like ion current'''
        return gbar * m**m_pow * h**h_pow * (e_ion)

    def g_e_k(self, n, gbar, n_pow, e_ion):
        '''calculates potasium-like ion current'''
        return gbar * n**n_pow * (e_ion)

    # condactivities
    def g_na(self, m, h, gbar, m_pow, h_pow):
        '''calculates sodium-like ion current'''
        return gbar * m**m_pow * h**h_pow

    def g_k(self, n, gbar, n_pow):
        '''calculates potasium-like ion current'''
        return gbar * n**n_pow


    def simulate(self, dt, t, i_inj, channels=False):
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
            # calculate sum of conductances multiplied by potentials
            g_e_nap = self.g_e_na(M_nap[n], H_nap[n], self.gbar_nap,
                              self.nap_m['pow'], self.nap_h['pow'], self.e_nap)
            g_e_nat = self.g_e_na(M_nat[n], H_nat[n], self.gbar_nat,
                              self.nat_m['pow'], self.nat_h['pow'], self.e_nat)
            g_e_hcn = self.g_e_k(N_hcn[n], self.gbar_hcn, self.hcn_n['pow'], self.e_hcn)
            g_e_kdr = self.g_e_k(N_kdr[n], self.gbar_kdr, self.kdr_n['pow'], self.e_kdr)

            g_e_leak = (self.g_leak) * (self.e_leak)
            g_e_sum = g_e_leak + (g_e_nap + g_e_nat + g_e_kdr + g_e_hcn)

            # calculate sum of conductances
            g_nap = self.g_na(M_nap[n], H_nap[n], self.gbar_nap, self.nap_m['pow'], self.nap_h['pow'])
            g_nat = self.g_na(M_nat[n], H_nat[n], self.gbar_nat, self.nat_m['pow'], self.nat_h['pow'])
            g_hcn = self.g_k(N_hcn[n], self.gbar_hcn, self.hcn_n['pow'])
            g_kdr = self.g_k(N_kdr[n], self.gbar_kdr, self.kdr_n['pow'])

            g_sum = (g_nap + g_nat + g_hcn + g_kdr + self.g_leak)


            # calculate membrane potential
            V_inf = (g_e_sum + i_inj[n] * 1e-3) / g_sum
            tau_v = (self.cm)* 1e-3  / g_sum

            U[n+1] = V_inf + (U[n] - V_inf) * np.exp(-dt / tau_v)


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
            M_nap[n+1] = self.dx_dt_exp(M_nap[n], M_nap_inf, tau_m_nap, dt)
            M_nat[n+1] = self.dx_dt_exp(M_nat[n], M_nat_inf, tau_m_nat, dt)
            H_nap[n+1] = self.dx_dt_exp(H_nap[n], H_nap_inf, tau_h_nap, dt)
            H_nat[n+1] = self.dx_dt_exp(H_nat[n], H_nat_inf, tau_h_nat, dt)
            N_hcn[n+1] = self.dx_dt_exp(N_hcn[n], N_hcn_inf, tau_n_hcn, dt)
            N_kdr[n+1] = self.dx_dt_exp(N_kdr[n], N_kdr_inf, tau_n_kdr, dt)


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
