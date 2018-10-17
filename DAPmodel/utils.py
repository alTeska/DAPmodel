import numpy as np
from DAPmodel import DAPSimulator


def obs_params(reduced_model=False):
    """Parameters for x_o
    Returns
    -------
    true_params : array
    labels_params : list of str
    """
    gbar_kdr = 0.00313  # (S/cm2)
    gbar_hcn = 5e-05    # (S/cm2)
    gbar_nap = 0.01527  # (S/cm2)
    gbar_nat = 0.142    # (S/cm2)

    true_params = np.array([gbar_kdr, gbar_hcn, gbar_nap, gbar_nat])
    labels_params = ['gbar_kdr', 'gbar_hcn', 'gbar_nap', 'gbar_nat']

    return true_params, labels_params


def syn_current(duration=300, dt=0.01, seed=None):
    """Simulation of triangular current"""
    l = duration/dt
    t = np.linspace(0, duration, int(l))

    I = np.zeros_like(t)
    i_up = np.linspace(0,3.5,250)
    i_down = np.linspace(3.5,0,250)
    I[15000:15500] = np.append(i_up, i_down)[:]

    return I, t

def syn_obs_data(I, dt, params, V0=-75, seed=None):
    """Data for x_o"""
    m = DAPSimulator(I=I, dt=dt, V0=V0, seed=seed)
    return m.gen_single(params)
