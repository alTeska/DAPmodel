from __future__ import print_function
import abc
import numpy as np
cimport numpy as np
from libc.math cimport exp


###################PARAMETERS#############################


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


cdef x_inf(double V, double x_vh, double x_vs):
    '''steady state values'''
    return 1 / (1 + np.exp((x_vh - V) / x_vs))

cdef i_na(double V, double m, double h, double gbar, double m_pow, double h_pow, double e_ion):
    '''calculates sodium-like ion current'''
    print('i work')
    return gbar * m**m_pow * h**h_pow * (V - e_ion)

def update_inf(double i):
    j = x_inf(1,2,3)

    cur = i_na(2, 1, 1, gbar_kdr, nap_m['pow'], nap_h['pow'], 1)

    print('cur:', cur)
    return cur, i+j
