import numpy as np

class Hodgkin_Huxley_Model():
    """Hodgkin Huxley Model for the tests of LFI"""

    # membrane capacitance, in uF/cm^2
    C_m = 1
    # maximum conducances, in mS/cm^2
    g_Na = 120.0
    g_K  = 36.0
    g_L  = 0.3
    # Nernst reversal potentials, in mV
    V_Na = 115
    V_K  = -12
    V_L  = 10.6
    V_rest = 0

    def __init__(self, temp):
        self.k = self.temp_corr(temp)

    def alpha_m(self, u): return (2.5 - 0.1 * u) / (np.exp(2.5 - 0.1 * u) - 1)
    def alpha_n(self, u): return (0.1 - 0.01 * u) / (np.exp(1 - 0.1 * u ) - 1)
    def alpha_h(self, u): return (0.07 * np.exp(-u / 20))

    def beta_m(self, u):  return 4 * np.exp((-u) / 18)
    def beta_n(self, u):  return 0.125 * np.exp((-u) / 80)
    def beta_h(self, u):  return 1 / (np.exp(3 - 0.1 * u) + 1)

    def x_inf(self, alpha, beta):
        '''Stedy State Values'''
        return alpha / (alpha + beta)

    def temp_corr(self, temp):
        '''temperature correction'''
        return 3**(0.1*(temp-6.3))

    def tau(self, alpha, beta):
        '''time constant calculation'''
        return  1 / (alpha + beta) * self.k


    def exp_euler_step(self, x, alpha, beta, dt):
        A = -(alpha + beta) * self.k
        B = alpha * self.k

        return x * np.exp(A*dt) + (B / A) * (np.exp(A*dt) - 1)


    def ionic_currents(self, M, N, H, u):
        '''Function calcualtes the hodgking huxley ionic currents, returns the sum of INa, IK and IL'''
        INa = self.g_Na * (M**3 * H) * (u - self.V_Na)  # sodium
        IK  = self.g_K  * (N**4    ) * (u - self.V_K)   # potas
        IL  = self.g_L               * (u - self.V_L)   # leakage

        return INa + IK + IL


    def simulate(self, T, dt, I_stim):
        ''' simluate hh model with given stimulation current, simulation done with exp euler method, returns '''
        t = np.linspace(0, T, int(T/dt))
        M = np.zeros_like(t)
        N = np.zeros_like(t)
        H = np.zeros_like(t)
        U = np.zeros_like(t)

        u0 = 0
        M[0] = self.x_inf(self.alpha_m(u0), self.beta_m(u0))
        N[0] = self.x_inf(self.alpha_n(u0), self.beta_n(u0))
        H[0] = self.x_inf(self.alpha_h(u0), self.beta_h(u0))

        for n in range(0, len(I_stim)-1):
            M[n+1] = self.exp_euler_step(M[n], self.alpha_m(U[n]), self.beta_m(U[n]), dt)
            N[n+1] = self.exp_euler_step(N[n], self.alpha_n(U[n]), self.beta_n(U[n]), dt)
            H[n+1] = self.exp_euler_step(H[n], self.alpha_h(U[n]), self.beta_h(U[n]), dt)

            I_ion = self.ionic_currents(M[n+1], N[n+1], H[n+1], U[n])
            U[n+1] = U[n] - 1/self.C_m * (I_ion - I_stim[n]) * dt

        return U, M, N, H
