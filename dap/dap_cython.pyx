from __future__ import print_function
import numpy as np
cimport numpy as np
import scipy
from libc.math cimport exp, log, cos, sqrt
cimport cython

###################PARAMETERS#############################

cdef double noise_fact = 0.5

cdef struct channel:
    int pow
    float vs # mV
    float vh # mV
    float tau_min # ms
    float tau_max # ms
    float tau_delta # ms

nap_m = channel(pow=3, vs= 16.11, vh=-52.82, tau_min=0.036, tau_max=15.332 , tau_delta=0.505)
nap_h = channel(pow=1, vs=-19.19, vh=-82.54, tau_min=0.336, tau_max=13.659 , tau_delta=0.439)
kdr_n = channel(pow=4, vh=-68.29, vs= 18.84, tau_min=0.286, tau_max=21.286 , tau_delta=0.746)
hcn_n = channel(pow=1, vh=-77.9 , vs=-20.54, tau_min=2.206, tau_max=137.799, tau_delta=0.21)
nat_m = channel(pow=3, vs= 11.99, vh=-30.94, tau_min=0    , tau_max=0.193  , tau_delta=0.187)
nat_h = channel(pow=1, vs=-13.17, vh=-60.44, tau_min=0.001, tau_max=8.743  , tau_delta=0.44)


cdef double gbar_kdr = 0.00313  # (S/cm2)
cdef double gbar_hcn = 5e-05    # (S/cm2)
cdef double gbar_nap = 0.01527  # (S/cm2)
cdef double gbar_nat = 0.142    # (S/cm2)

cdef double cm = 0.63      #* 1e-6  # uF/cm2
cdef double diam = 50.0    * 1e-4  # cm
cdef double L = 100.0      * 1e-4  # cm
cdef double Ra = 100.0
cdef double cell_area = diam * L * np.pi

cm = cm * cell_area  # uF
gbar_kdr = gbar_kdr * cell_area  # S
gbar_hcn = gbar_hcn * cell_area  # S
gbar_nap = gbar_nap * cell_area  # S
gbar_nat = gbar_nat * cell_area  # S

cdef double e_hcn = -29.46    # mV
cdef double e_nap = 60.0      # mV
cdef double e_nat = 60.0      # mV
cdef double e_leak = -86.53   # mV
cdef double e_kdr = -110.0    # mV
cdef double g_leak = 0.000430
g_leak = g_leak * cell_area

############################################################

def setnoisefactor(double x):
    """ Changes the noise_factor to the one set in python interface"""
    global noise_fact
    noise_fact = x

def setparams(params):
    '''
    Function used to set up the expected parameters, expected lenghts
    are range from 1 to 5 and can be extended to 11.
    '''
    global gbar_nap, nap_m, nap_h, nat_m, nat_h, gbar_leak, gbar_nat, gbar_kdr, gbar_hcn
    params_len = np.size(params)

    if params_len == 1:
        gbar_nap = params[0] * 0.001 * cell_area
    elif params_len == 2:
        gbar_nap = params[0] * 0.001 * cell_area
        gbar_leak = params[1] * 0.001 * cell_area
    elif params_len == 3:
        gbar_nap = params[0] * 0.001 * cell_area
        gbar_leak = params[1] * 0.001 * cell_area
        gbar_nat = params[2] * 0.001 * cell_area
    elif params_len == 4:
        gbar_nap = params[0] * 0.001 * cell_area
        gbar_leak = params[1] * 0.001 * cell_area
        gbar_nat = params[2] * 0.001 * cell_area
        gbar_kdr = params[3] * 0.001 * cell_area
    elif params_len == 5:
        gbar_nap = params[0] * 0.001 * cell_area
        gbar_leak = params[1] * 0.001 * cell_area
        gbar_nat = params[2] * 0.001 * cell_area
        gbar_kdr = params[3] * 0.001 * cell_area
        gbar_hcn = params[4] * 0.001 * cell_area
    elif params_len == 11:
        gbar_nap = params[0] * 0.001 * cell_area
        gbar_leak = params[1] * 0.001 * cell_area
        gbar_nat = params[2] * 0.001 * cell_area
        gbar_kdr = params[3] * 0.001 * cell_area
        gbar_hcn = params[4] * 0.001 * cell_area
        nap_h['tau_max'] = params[5]
        nap_h['vs'] = params[6]
        nap_m['tau_max'] = params[7]
        nap_m['vs'] = params[8]
        kdr_n['tau_max'] = params[9]
        kdr_n['vs'] = params[10]
    else:
        raise ValueError('You can only provide 1, 2, 3, 4, 5 or 11 parameters!')


# model integration
@cython.cdivision(True)
cdef double x_inf(double V, double x_vh, double x_vs):
    '''steady state values'''
    return 1 / (1 + np.exp((x_vh - V) / x_vs))

@cython.cdivision(True)
cdef double x_tau(double V, double xinf, ion_ch):
    return (ion_ch['tau_min'] + (ion_ch['tau_max'] - ion_ch['tau_min']) * \
            xinf * np.exp(ion_ch['tau_delta'] * \
            (ion_ch['vh'] - V) / ion_ch['vs']))

# currents
cdef double i_na(double V, double m, double h, double gbar, double m_pow, double h_pow, double e_ion):
    '''calculates sodium-like ion current'''
    return gbar * m**m_pow * h**h_pow * (V - e_ion)

cdef double i_k(double V, double n, double gbar, double n_pow, double e_ion):
    '''calculates potasium-like ion current'''
    return gbar * n**n_pow * (V - e_ion)

@cython.cdivision(True)
cdef double dx_dt(double x, double x_inf, double x_tau):
    '''differential equations for m,h,n'''
    return (x_inf - x) / x_tau


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void udapte_forwardeuler(np.ndarray[double,ndim=1] i_inj, np.ndarray[double,ndim=1] U, np.ndarray[double,ndim=1] M_nap, np.ndarray[double,ndim=1] M_nat,np.ndarray[double,ndim=1] H_nap, np.ndarray[double,ndim=1] H_nat, np.ndarray[double,ndim=1] N_hcn, np.ndarray[double,ndim=1] N_kdr, double dt, np.ndarray[double,ndim=1] r_mat, int n):
    """
    Function updates all of the activation parameters and voltage of each iteration of DAP model forward integration.
    """

    cdef double u = U[n-1]
    cdef double m_nap = M_nap[n-1]
    cdef double m_nat = M_nat[n-1]
    cdef double h_nap = H_nap[n-1]
    cdef double h_nat = H_nat[n-1]
    cdef double n_hcn = N_hcn[n-1]
    cdef double n_kdr = N_kdr[n-1]

    cdef m_nap_inf, m_nat_inf, h_nap_inf, h_nat_inf, n_hcn_inf, n_kdr_inf
    cdef tau_m_nap, tau_h_nap, tau_m_nat, tau_h_nat, tau_n_hcn, tau_n_kdr

    m_nap_inf = x_inf(u, nap_m['vh'], nap_m['vs'])
    m_nat_inf = x_inf(u, nat_m['vh'], nat_m['vs'])
    h_nap_inf = x_inf(u, nap_h['vh'], nap_h['vs'])
    h_nat_inf = x_inf(u, nat_h['vh'], nat_h['vs'])
    n_hcn_inf = x_inf(u, hcn_n['vh'], hcn_n['vs'])
    n_kdr_inf = x_inf(u, kdr_n['vh'], kdr_n['vs'])

    # calcualte  time constants
    tau_m_nap = x_tau(u, m_nap_inf, nap_m)
    tau_h_nap = x_tau(u, h_nap_inf, nap_h)
    tau_m_nat = x_tau(u, m_nat_inf, nat_m)
    tau_h_nat = x_tau(u, h_nat_inf, nat_h)
    tau_n_hcn = x_tau(u, n_hcn_inf, hcn_n)
    tau_n_kdr = x_tau(u, n_kdr_inf, kdr_n)

    # calculate all steady states
    m_nap = m_nap + dx_dt(m_nap, m_nap_inf, tau_m_nap) * dt
    m_nat = m_nat + dx_dt(m_nat, m_nat_inf, tau_m_nat) * dt
    h_nap = h_nap + dx_dt(h_nap, h_nap_inf, tau_h_nap) * dt
    h_nat = h_nat + dx_dt(h_nat, h_nat_inf, tau_h_nat) * dt
    n_hcn = n_hcn + dx_dt(n_hcn, n_hcn_inf, tau_n_hcn) * dt
    n_kdr = n_kdr + dx_dt(n_kdr, n_kdr_inf, tau_n_kdr) * dt


    # calculate ionic currents
    i_nap = i_na(u, m_nap, h_nap, gbar_nap, nap_m['pow'], nap_h['pow'], e_nap)
    i_nat = i_na(u, m_nat, h_nat, gbar_nat, nat_m['pow'], nat_h['pow'], e_nat)
    i_hcn = i_k(u, n_hcn, gbar_hcn, hcn_n['pow'], e_hcn)
    i_kdr = i_k(u, n_kdr, gbar_kdr, kdr_n['pow'], e_kdr)

    i_ion = (i_nap + i_nat + i_kdr + i_hcn) * 1e3
    i_leak = (g_leak) * (u - e_leak) * 1e3

    # calculate membrane potential
    u = u + (-i_ion - i_leak + i_inj[n-1])/(cm) * dt

    U[n] = u + r_mat[n] * noise_fact
    M_nap[n] = m_nap
    M_nat[n] = m_nat
    H_nap[n] = h_nap
    H_nat[n] = h_nat
    N_hcn[n] = n_hcn
    N_kdr[n] = n_kdr



# python based functions
def forwardeuler(np.ndarray[double,ndim=1] t, np.ndarray[double,ndim=1] I, np.ndarray[double,ndim=1] U, np.ndarray[double,ndim=1] M_nap, np.ndarray[double,ndim=1] M_nat, np.ndarray[double,ndim=1] H_nap, np.ndarray[double,ndim=1] H_nat, np.ndarray[double,ndim=1] N_hcn, np.ndarray[double,ndim=1] N_kdr, double dt, np.ndarray[double,ndim=1] r_mat):
    """
    Function initiates the values required to go through DAP forward integration.
    Then iterates through the required duration.
    """

    M_nap[0] = x_inf(U[0], nap_m['vh'], nap_m['vs'])
    M_nat[0] = x_inf(U[0], nat_m['vh'], nat_m['vs'])
    H_nap[0] = x_inf(U[0], nap_h['vh'], nap_h['vs'])
    H_nat[0] = x_inf(U[0], nat_h['vh'], nat_h['vs'])
    N_hcn[0] = x_inf(U[0], hcn_n['vh'], hcn_n['vs'])
    N_kdr[0] = x_inf(U[0], kdr_n['vh'], kdr_n['vs'])

    for n in range(1, t.shape[0]):
        udapte_forwardeuler(I, U, M_nap, M_nat, H_nap, H_nat, N_hcn, N_kdr, dt, r_mat, n)
