import numpy as np
from .DAPbase import DAPBase


class DAPBe(DAPBase):
    """
    DAP Model based on HH equations for the tests with LFI
    Model conists of 4 types of ion channels: Nap, Nat, Kdr, hcn_slow.
    i_inj = nA

    Integrated with Backward(implicit) Euler method.
    """

    def __init__(self, state, params, seed=None, **kwargs):
        super().__init__(state=state, params=params,
                         seed=seed, **kwargs)



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

    # condactivities
    def g_na(self, m, h, gbar, m_pow, h_pow):
        '''calculates sodium-like ion current'''
        return gbar * m**m_pow * h**h_pow

    def g_k(self, n, gbar, n_pow):
        '''calculates potasium-like ion current'''
        return gbar * n**n_pow


    def func(self, V_new, V_old, i_ion, i_leak, i_inj, cm,  dt):
        """ The function f(x) we want the root of."""
        return V_new - V_old - dt*(-i_ion - i_leak + i_inj)/cm

    def dfuncdx(self, g_sum, dt):
        """ The derivative of f(x) with respect to x_new."""
        return 1 + dt * g_sum


    def newton(self, V_old, i_ion, i_leak, i_inj, cm, g_sum, dt, precision=1e-12):
        """ Takes values x_old and t_new, and finds the root of the
        function f(x_new), returning x_new. """

        # initial guess:
        V_new = V_old
        f = self.func(V_new, V_old, i_ion, i_leak, i_inj, cm, dt)
        dfdx = self.dfuncdx(g_sum, dt)

        # update guess, till desired precisions:
        while abs(f/dfdx) > precision:
            V_new = V_new - f/dfdx
            f = self.func(V_new, V_old, i_ion, i_leak, i_inj, cm, dt)
            dfdx = self.dfuncdx(g_sum, dt)

        return V_new

    def x_func(self, x_new, x_old, x_inf, tau_x, dt):
        """ The function f(x) we want the root of."""
        return x_new - x_old - dt*(x_inf - x_new)/tau_x

    def x_dfuncdx(self, tau_x, dt):
        """ The derivative of f(x) with respect to V_new."""
        return 1 + dt/tau_x

    def x_newton(self, x_old, x_inf, tau_x, dt, precision=1e-12):
        """ Takes values x_old and t_new, and finds the root of the
        function f(x_new), returning x_new. """

        # initial guess:
        x_new = x_old
        f = self.x_func(x_new, x_old, x_inf, tau_x, dt)
        dfdx = self.x_dfuncdx(tau_x, dt)

        while abs(f/dfdx) > precision:
            x_new = x_new - f/dfdx
            f = self.x_func(x_new, x_old, x_inf, tau_x, dt)
            dfdx = self.x_dfuncdx(tau_x, dt)

        return x_new


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
            # self.x_newton(x_old, x_inf, tau_x, dt)
            M_nap[n+1] = self.x_newton(M_nap[n], M_nap_inf, tau_m_nap, dt)
            M_nat[n+1] = self.x_newton(M_nat[n], M_nat_inf, tau_m_nat, dt)
            H_nap[n+1] = self.x_newton(H_nap[n], H_nap_inf, tau_h_nap, dt)
            H_nat[n+1] = self.x_newton(H_nat[n], H_nat_inf, tau_h_nat, dt)
            N_hcn[n+1] = self.x_newton(N_hcn[n], N_hcn_inf, tau_n_hcn, dt)
            N_kdr[n+1] = self.x_newton(N_kdr[n], N_kdr_inf, tau_n_kdr, dt)


            # calculate sum of conductances
            g_nap = self.g_na(M_nap[n], H_nap[n], self.gbar_nap, self.nap_m['pow'], self.nap_h['pow'])
            g_nat = self.g_na(M_nat[n], H_nat[n], self.gbar_nat, self.nat_m['pow'], self.nat_h['pow'])
            g_hcn = self.g_k(N_hcn[n], self.gbar_hcn, self.hcn_n['pow'])
            g_kdr = self.g_k(N_kdr[n], self.gbar_kdr, self.kdr_n['pow'])

            g_sum = (g_nap + g_nat + g_hcn + g_kdr)


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
            U[n+1] = self.newton(U[n], i_ion, i_leak, i_inj[n], self.cm, g_sum, dt)



        # return U.reshape(-1,1) #+ nois_fact_obs*self.rng.randn(t.shape[0],1)
        return U.reshape(-1,1), M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr
