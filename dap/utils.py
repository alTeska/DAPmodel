# model based on lfimodels library by Jan-Matthis Lückmann
import numpy as np
import delfi.distribution as dd
from delfi.summarystats import Identity

from dap import DAP, DAPBe
from .dap_simulator import DAPSimulator
from .dap_sumstats_dict import DAPSummaryStatsDict
from .dap_sumstats import DAPSummaryStats

# from lfimodels.hodgkinhuxley.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments
# from lfimodels.hodgkinhuxley.HodgkinHuxleyStatsSpikes import HodgkinHuxleyStatsSpikes
# from lfimodels.hodgkinhuxley.HodgkinHuxleyStatsSpikes_mf import HodgkinHuxleyStatsSpikes_mf
from delfi.summarystats import Identity



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
        true_params = np.array([gbar_nap])
        labels_params = ['gbar_nap']

    else:
        true_params = np.array([gbar_nap, gbar_leak, gbar_nat, gbar_kdr, gbar_hcn])
        labels_params = ['gbar_nap', 'gbar_leak', 'gbar_nat', 'gbar_kdr', 'gbar_hcn']


    return true_params, labels_params


def obs_params(reduced_model=True):
    """
    Parameters for x_o, two optionss: either 2 params (reduced_model=True) or 10
    Returns
    -------
    params : array
    labels : list of str

    """

    if reduced_model:
        params = np.zeros(2)
        params[0] = 0.01527  # gbar_nap
        params[1] = 16.11    # nap_m['vs']
        labels = ['gbar_nap', 'nap_m_vs']
    else:
        params = np.zeros(10)
        params[0] = 0.01527  # gbar_nap
        params[1] = 16.11    # nap_m['vs']
        params[2] = 15.332   # nap_m['tau_max']
        params[3] =-19.19    # nap_h['vs']
        params[4] = 13.659   # nap_h['tau_max']
        params[5] =-30.94    # nat_m['vh']
        params[6] =-60.44    # nat_h['vh']
        params[7] = 11.99    # nat_m['vs']
        params[8] =-13.17    # nat_h['vs']
        params[9] = 18.84    # kdr_n['vs']
        labels = ['gbar_nap', 'nap_m_vs', 'nap_m_tau_max', 'nap_h_vs',
                  'nap_h_tau_max', 'nat_m_vh', 'nat_h_vh', 'nat_m_vs',
                  'nat_h_vs', 'krd_n']


    return params, labels


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
        s = DAPSummaryStatsDict(t_on, t_off, n_summary=n_summary)
    elif summary_stats == 1:
        s = DAPSummaryStats(t_on, t_off, n_summary=n_summary)
    else:
        raise ValueError('Only 0, 1 as an option for summary statistics.')
    # elif summary_stats == 2:
    #     s = HodgkinHuxleyStatsMoments(t_on, t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary)

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
