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

    # condactivities
    def g_na(self, m, h, gbar, m_pow, h_pow):
        '''calculates sodium-like ion current'''
        return gbar * m**m_pow * h**h_pow

    def g_k(self, n, gbar, n_pow):
        '''calculates potasium-like ion current'''
        return gbar * n**n_pow

    # integration
    def func(self, old, new, inf, tau, dt):
        """ The function f(x) we want the root of steady states."""
        return new - old - dt*(inf - new)/tau

    def dfuncdx(self, tau, dt):
        """ The derivative of f(x) with respect to steady state calculation [dfucn/dt]"""
        return 1 + dt/tau

    def Ufunc(self, old, new, i_ion, i_leak, i_inj, g_sum, dt):
        """ The function f(x) we want the root of the voltage (U)."""
        return new - old - dt*(-i_ion - i_leak + i_inj)/self.cm

    def dUfuncdx(self, g_sum, dt):
        """ The derivative of f(x) with respect to voltage (U) calculations.[dUfucn/dt]"""
        return 1 + dt * g_sum

    def newton_uni(self, old, func, dfuncdx, *args):
        """
        Function solves the problem of same value in implicit Euler, by calculation of newton roots.
        Takes values old and t_new, and finds the root of the function f(new), returns new.
        Looks for given precision of new. """

        # initial guess:
        new = old
        f = func(old, new, *args)
        dfdx = dfuncdx(args[-2], args[-1])

        # update guess, till desired precisions:
        while abs(f/dfdx) > self.precision:
            new = new - f/dfdx
            f = func(old, new, *args)
            dfdx = dfuncdx(args[-2], args[-1])

        return new

    def simulate(self, dt, t, i_inj, channels=False, noise=False, noise_fact=1e-3, precision=1e-12):
        """Run simulation of DAP model given the injection current

        Parameters
        ----------
        dt (float): Timestep
        t  (array): array with time course
        i_inj (array): array with the input I
        channels (bool): decides if activation channels should be returned
        noise (bool): decides about adding noise to the voltage trace
        noise_fact (float): size of the added noise

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
        self.precision = precision
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

            # calculate steady states
            M_nap[n+1] = self.newton_uni(M_nap[n], self.func, self.dfuncdx, M_nap_inf, tau_m_nap, dt)
            M_nat[n+1] = self.newton_uni(M_nat[n], self.func, self.dfuncdx, M_nat_inf, tau_m_nat, dt)
            H_nap[n+1] = self.newton_uni(H_nap[n], self.func, self.dfuncdx, H_nap_inf, tau_h_nap, dt)
            H_nat[n+1] = self.newton_uni(H_nat[n], self.func, self.dfuncdx, H_nat_inf, tau_h_nat, dt)
            N_hcn[n+1] = self.newton_uni(N_hcn[n], self.func, self.dfuncdx, N_hcn_inf, tau_n_hcn, dt)
            N_kdr[n+1] = self.newton_uni(N_kdr[n], self.func, self.dfuncdx, N_kdr_inf, tau_n_kdr, dt)


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
            U[n+1] = self.newton_uni(U[n], self.Ufunc, self.dUfuncdx, i_ion,
                                     i_leak, i_inj[n], g_sum, dt)


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
