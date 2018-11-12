# model based on lfimodels library by Jan-Matthis Lückmann
import numpy as np
import delfi.distribution as dd
from delfi.summarystats import Identity

from DAPmodel import DAP, DAPSimulator
from .DAP_sumstats import DAPSummaryStats

from lfimodels.hodgkinhuxley.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments
from lfimodels.hodgkinhuxley.HodgkinHuxleyStatsSpikes import HodgkinHuxleyStatsSpikes
from lfimodels.hodgkinhuxley.HodgkinHuxleyStatsSpikes_mf import HodgkinHuxleyStatsSpikes_mf
from delfi.summarystats import Identity



def obs_params_gbar(reduced_model=False):
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

def obs_params():
    """Parameters for x_o
    Returns
    -------
    true_params : array
    labels_params : list of str
    """
    # high variability parameters
    nap_m_tau_max = 15.332   # ms
    nap_m_vs = 16.11         # mV
    nap_h_tau_max = 13.659   # ms

    # medium variability
    nap_h_tau_delta = 0.439    # ms

    # true_params = np.array([nap_m_tau_max, nap_m_vs, nap_h_tau_max])
    # labels_params = ['nap_m_tau_max', 'nap_m_vs', 'nap_h_tau_max']

    true_params = np.array([nap_m_tau_max, nap_h_tau_delta])
    labels_params = ['nap_m_tau_max', 'nap_h_tau_delta']

    return true_params, labels_params



def syn_current(duration=200, dt=0.01, t_on=55, t_off=60, amp=3.1, seed=None, on_off=False):
    """Simulation of triangular current"""
    t = np.arange(0, duration+dt, dt)
    I = np.zeros_like(t)

    stim = len(I[int(np.round(t_on/dt)):int(np.round(t_off/dt))])

    i_up = np.linspace(0, amp, (stim/2))
    i_down = np.linspace(amp, 0, (stim/2))

    I[int(np.round(t_on/dt)):int(np.round(t_off/dt))] = np.append(i_up, i_down)[:]

    return I, t, t_on, t_off


def syn_obs_data(I, dt, params, V0=-75, seed=None):
    """Data for x_o"""
    m = DAPSimulator(I=I, dt=dt, V0=V0, seed=seed)
    return m.gen_single(params)


def syn_obs_stats(I, params, dt, t_on, t_off, data=None, V0=-75, summary_stats=1, n_xcorr=5,
                  n_mom=5, n_summary=4, seed=None):
    """Summary stats for x_o of DAP"""

    if data is None:
        m = DAP(I=I, dt=dt, V0=V0, seed=seed)
        data = m.gen_single(params)

    if summary_stats == 0:
        s = Identity()
    elif summary_stats == 1:
        s = HodgkinHuxleyStatsMoments(t_on, t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary)
    elif summary_stats == 2:
        s = HodgkinHuxleyStatsSpikes(t_on, t_off, n_summary=n_summary)
    elif summary_stats == 3:
        s = HodgkinHuxleyStatsSpikes_mf(t_on, t_off, n_summary=n_summary)
    elif summary_stats == 4:
        s = DAPSummaryStats(t_on, t_off, n_summary=n_summary)
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
