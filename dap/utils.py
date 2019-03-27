# model based on lfimodels library by Jan-Matthis Lückmann
import numpy as np
import delfi.distribution as dd

from dap import DAPcython
from .dap_simulator import DAPSimulator
from .dap_sumstats_moments import DAPSummaryStatsMoments
from .dap_sumstats_step_mom import DAPSummaryStatsStepMoments
from dap.cell_fitting.read_heka import (get_sweep_index_for_amp, shift_v_rest,
                                        get_i_inj_from_function,
                                        get_v_and_t_from_heka)


def load_current(data_dir, protocol='rampIV', ramp_amp=3.1):
    '''
    Loads the current from recorded dataset.abs

    protocol: 'rampIV', 'IV', 'Zap20'
    ramp_amp:   steps of 0.05 -0.15

    * ramp_amp for rampIV=3.1, ramp_amp for 'IV'=1
    '''
    v_shift = -16  # shift for accounting for the liquid junction potential

    if protocol == 'Zap20':
        sweep_idx = 0
    else:
        sweep_idx = get_sweep_index_for_amp(ramp_amp, protocol)

    v, t = get_v_and_t_from_heka(data_dir, protocol, sweep_idxs=[sweep_idx])
    v = shift_v_rest(v[0], v_shift)
    t = t[0]
    dt = t[1] - t[0]

    I, t_on, t_off = get_i_inj_from_function(protocol, [sweep_idx], t[-1], dt,
                                             return_discontinuities=False)

    return I[0], v, t, t_on, t_off, dt


def obs_params_gbar(reduced_model=True):
    """Parameters for x_o
    Returns
    -------
    true_params : array
    labels_params : list of str
    """
    gbar_nap = 0.01527    # (S/cm2)
    gbar_leak = 0.000430  # (S/cm2)
    gbar_nat = 0.142      # (S/cm2)
    gbar_kdr = 0.00313    # (S/cm2)
    gbar_hcn = 5e-05      # (S/cm2)

    if reduced_model:
        true_params = np.array([gbar_nap, gbar_leak])
        labels_params = ['gbar_nap', 'gbar_leak']

    else:
        true_params = np.array([gbar_nap, gbar_leak, gbar_nat, gbar_kdr, gbar_hcn])
        labels_params = ['gbar_nap', 'gbar_leak', 'gbar_nat', 'gbar_kdr', 'gbar_hcn']

    return true_params*10, labels_params

def obs_params(reduced_model=False):

    """
    Parameters for x_o, two optionss: either 2 params (reduced_model=True) or 10
    Returns
    -------
    params : array
    labels : list of str

    """

    if reduced_model:
        params = np.zeros(5)
        params[0] = 0.01527  * 10  # (S/cm2)
        params[1] = 0.000430 * 10  # (S/cm2)
        params[2] = 0.142    * 10  # (S/cm2)
        params[3] = 0.00313  * 10  # (S/cm2)
        params[4] = 5e-05    * 10  # (S/cm2)

        labels = ['gbar_nap', 'gbar_leak', 'gbar_nat', 'gbar_kdr', 'gbar_hcn']
    else:
        params = np.zeros(11)
        params[0] = 0.01527  * 10  # (S/cm2)
        params[1] = 0.000430 * 10  # (S/cm2)
        params[3] = 0.00313  * 10  # (S/cm2)
        params[2] = 0.142    * 10  # (S/cm2)
        params[4] = 5e-05    * 10  # (S/cm2)

        params[5] = 13.659   # nap_h['tau_max']
        params[6] =-19.19    # nap_h['vs']
        params[7] = 15.332   # nap_m['tau_max']
        params[8] = 16.11    # nap_m['vs']
        params[9] = 21.286   # kdr_n['tau_max']
        params[10] = 18.84   # kdr_n['vs']
        labels = ['gbar_nap', 'gbar_leak', 'gbar_nat', 'gbar_kdr', 'gbar_hcn',
                  'nap_h_tau_max', 'nap_h_vs', 'nap_m_tau_max', 'nap_m_vs',
                  'kdr_n_tau_max', 'kdr_n_vs']

    return params, labels

def load_prior_ranges():
    """Returns ranges of parameters narrowed down based on best 3500 models"""

    labels = ['gbar_nap', 'gbar_leak', 'gbar_nat', 'gbar_kdr', 'gbar_hcn',
              'nap_h_tau_max', 'nap_h_vs', 'nap_m_tau_max', 'nap_m_vs',
              'kdr_n_tau_max', 'kdr_n_vs']

    prior_min = np.array((0.003274, 0.001, 0.021962, 0.001925, 0.000041,
                          1.662074, -31.186329, 3.384709, 4.124004, 9.287113,
                          6.848391))


    prior_max = np.array((0.027263, 0.02, 0.261845, 0.004325, 0.000065,
                      25.651677, -7.194925, 27.320601, 28.028747, 33.284546))

    return prior_min, prior_max, labels


def syn_current(duration=200, dt=0.01, t_on=55, t_off=60, amp=3.1, seed=None, on_off=False):
    """Simulation of triangular current"""
    t = np.arange(0, duration, dt)
    I = np.zeros_like(t)

    stim = len(I[int(np.round(t_on/dt)):int(np.floor(t_off/dt))])

    i_up = np.linspace(0, amp, (stim/2))
    i_down = np.linspace(amp, 0, (stim/2))

    I[int(np.round(t_on/dt)):int(np.round(t_off/dt))] = np.append(i_up, i_down)[:]

    return I, t, t_on, t_off


def syn_obs_data(I, dt, params, V0=-75, seed=None):
    """Data for x_o"""
    m = DAPSimulator(I=I, dt=dt, V0=V0, seed=seed)
    return m.gen_single(params)


def syn_obs_stats(I, params, dt, t_on, t_off, data=None, V0=-75, summary_stats=0, n_xcorr=5,
                  n_mom=5, n_summary=4, seed=None):
    """Summary stats for x_o of DAP"""

    if data is None:
        m = DAPcython(I=I, dt=dt, V0=V0, seed=seed)
        data = m.gen_single(params)

    if summary_stats == 0:
        s = DAPSummaryStatsMoments(t_on, t_off, n_summary=n_summary)
    elif summary_stats == 1:
        s = DAPSummaryStatsStepMoments(t_on, t_off, n_summary=n_summary)
    else:
        raise ValueError('Only 0, 1 as an option for summary statistics.')

    return s.calc([data])


def prior(true_params, seed=None, prior_log=False, prior_uniform=False):
    """Prior"""
    range_lower = param_transform(prior_log ,0.5*true_params)
    range_upper = param_transform(prior_log, 1.5*true_params)

    range_lower = range_lower[0:len(true_params)]
    range_upper = range_upper[0:len(true_params)]

    if prior_uniform:
        prior_min = range_lower
        prior_max = range_upper

        return dd.Uniform(lower=prior_min, upper=prior_max,
                               seed=seed)
    else:
        prior_mn = param_transform(prior_log,true_params)
        prior_cov = np.diag((range_upper - range_lower)**2)/12

        return dd.Gaussian(m=prior_mn, S=prior_cov, seed=seed)


def param_transform(prior_log, x):
    if prior_log:
        return np.log(x)
    else:
        return x


def param_invtransform(prior_log, x):
    if prior_log:
        return np.exp(x)
    else:
        return x
