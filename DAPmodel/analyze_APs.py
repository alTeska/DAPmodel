from __future__ import division
from scipy.signal import argrelmin, argrelmax
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from functools import partial
import numpy as np
import matplotlib.pyplot as pl

__author__ = 'caro'

def get_spike_characteristics_dict(for_data=False):
    spike_characteristics_dict = {
        'AP_threshold': -10,  # mV
        'AP_interval': 2.5,  # ms
        'fAHP_interval': 4.0,
        'AP_width_before_onset': 2.0,  # ms
        'DAP_interval': 10.0,  # ms
        'order_fAHP_min': 1.0,  # ms (how many points to consider for the minimum)
        'order_DAP_max': 1.0,  # ms (how many points to consider for the minimum)
        'min_dist_to_DAP_max': 0.5,  # ms
        'k_splines': 3,
        's_splines': 0  # 0 means no interpolation, use for models
    }
    if for_data:
        spike_characteristics_dict['s_splines'] = None
    return spike_characteristics_dict


def to_idx(time_point, dt, decimal_place=None):
    idx = time_point/dt
    if decimal_place:
        assert np.round(idx * dt, decimal_place) == np.round(time_point, decimal_place), \
            'Time points given are not uniquely identifiable given dt. ' \
            'time_point: %f, dt: %f' % (time_point, dt)
    else:
        assert idx * dt == time_point, 'Time points given are not uniquely identifiable given dt. ' \
                                       'time_point: %f, dt: %f' % (time_point, dt)
    return int(idx)


def exp_fit(t, a, v):
    diff_exp = np.max(np.exp(-t / a)) - np.min(np.exp(-t / a))
    diff_points = v[0] - v[-1]
    return (np.exp(-t / a) - (np.exp(-t / a))[0]) / diff_exp * diff_points + v[0]


def get_AP_onset_idxs(v, threshold=-30):
    """
    Returns the indices of the times where the membrane potential crossed threshold.
    :param threshold (float): AP threshold.
    :return (array_like): Indices of the times where the membrane potential crossed threshold.
    """
    return np.nonzero(np.diff(np.sign(v-threshold)) == 2)[0]


def get_AP_max_idx(v, AP_onset, AP_end, order=1, interval=None, v_diff_onset_max=None, add_noise=False):
    """
    Returns the index of the local maximum of the AP between AP onset and end during dur.
    :param AP_onset (int): Index where the membrane potential crosses the AP threshold.
    :param AP_end (int): Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace).
    :param order (int): Number of points to consider for determining the local maxima.
    :param interval (int): Length of the interval during which the maximum of the AP shall occur starting from AP onset.
    :return(int): Index of the Maximum of the AP (None if it does not exist).
    """
    if add_noise:
        maxima = argrelmax(v[AP_onset:AP_end] + np.random.uniform(-0.001, 0.001, len(v[AP_onset:AP_end])),
                                                                  order=order)[0]  # add noise so that if two
        # neighboring points are identical, one can be found as minimum
    else:
        maxima = argrelmax(v[AP_onset:AP_end], order=order)[0]
    if interval is not None:
        maxima = maxima[maxima < interval]

    if v_diff_onset_max is not None:
        maxima = maxima[(v[AP_onset + maxima] - v[AP_onset]) > v_diff_onset_max]

    if np.size(maxima) == 0:
        return None
    else:
        return maxima[np.argmax(v[AP_onset:AP_end][maxima])] + AP_onset


def get_fAHP_min_idx(v, AP_max, AP_end, order=1, interval=None):
    """
    Returns the index of the local minimum found after AP maximum.
    :param AP_max (int): Index of the maximum of the AP.
    :param AP_end (int): Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace).
    :param order (int): Number of points to consider for determining the local minima.
    :param interval (int): Length of the interval during which the minimum of the fAHP shall occur starting from AP max.
    :return (int): Index of the Minimum of the fAHP (None if it does not exist).
    """
    minima = argrelmin(v[AP_max:AP_end], order=order)[0]
    if interval is not None:
        minima = minima[minima < interval]

    if np.size(minima) == 0:
        return None
    else:
        return minima[np.argmin(v[AP_max:AP_end][minima])] + AP_max


def get_fAHP_min_idx_using_splines(v, t, AP_max_idx, AP_end, order=None, interval=None, w=None, s=None, k=3):

    if w is not None:
        w = w[AP_max_idx:AP_end]
    splines = UnivariateSpline(t[AP_max_idx:AP_end], v[AP_max_idx:AP_end], w=w, s=s, k=k)
    v_new = splines(t[AP_max_idx:AP_end])

    # import matplotlib.pyplot as pl
    # pl.figure()
    # pl.plot(t, v, 'k')
    # pl.plot(t[AP_max_idx:AP_end], v_new, 'r')
    # pl.show()

    fAHP_min_idx = get_fAHP_min_idx(v_new, 0, len(v_new), order, interval)
    if fAHP_min_idx is None:
        return None
    return fAHP_min_idx + AP_max_idx


def get_DAP_max_idx(v, fAHP_min, AP_end, order=None, interval=None, min_dist_to_max=None):
    """
    Returns the index of the local maximum found after fAHP.
    :param fAHP_min(int): Index of the minimum of the fAHP.
    :param AP_end (int): Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace).
    :param order (int): Number of points to consider for determining the local minima.
    :param interval (int): Length of the interval during which the minimum of the fAHP shall occur starting from AP max.
    :return (int): Index of maximum of the DAP (None if it does not exist).
    """
    maxima = argrelmax(v[fAHP_min:AP_end], order=order)[0]
    if interval is not None:
        maxima = maxima[maxima < interval]
    if min_dist_to_max is not None:
        maxima = maxima[maxima >= min_dist_to_max]

    if np.size(maxima) == 0:
        return None
    else:
        return maxima[np.argmax(v[fAHP_min:AP_end][maxima])] + fAHP_min


def get_DAP_max_idx_using_splines(v, t, fAHP_min, AP_end, order=None, interval=None, min_dist_to_max=None, w=None,
                                  s=None, k=3):
    splines = UnivariateSpline(t, v, w=w, s=s, k=k)
    v_new = splines(t[fAHP_min:AP_end])

    # import matplotlib.pyplot as pl
    # pl.figure()
    # pl.plot(t, v, 'k')
    # pl.plot(t[fAHP_min:AP_end], v_new, 'r')
    # pl.show()

    DAP_max_idx = get_DAP_max_idx(v_new, 0, len(v_new), order, interval, min_dist_to_max)
    if DAP_max_idx is None:
        return None
    return DAP_max_idx + fAHP_min


def get_AP_amp(v, AP_max, vrest):
    """
    Computes the amplitude of the AP in relation to the resting potential.
    :param AP_max (int): Index of the maximum of the AP.
    :param vrest (float): Resting potential.
    :return (float): Amplitude of the AP.
    """
    return v[AP_max] - vrest


def get_AP_width_idxs(v, t, AP_onset, AP_max, AP_end, vrest):
    """
    Computes the width at half maximum of the AP.
    :param AP_onset (int): Index where the membrane potential crosses the AP threshold.
    :param AP_max (int): Index of the maximum of the AP.
    :param AP_end (int): Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace).
    :param vrest (float): Resting potential.
    :return (Union[int, int]): Indices of where voltage crosses the AP half maximum.
    """
    halfmax = (v[AP_max] - vrest)/2

    if AP_onset < 0:
        AP_onset = 0
    start_idx = np.nonzero(np.diff(np.sign(v[AP_onset:AP_max]-vrest-halfmax)) == 2)[0]
    if len(start_idx) > 0:
        start_idx = start_idx[0] + AP_onset
    else:
        start_idx = None
    end_idx = np.nonzero(np.diff(np.sign(v[AP_max:AP_end]-vrest-halfmax)) == -2)[0]
    if len(end_idx) > 0:
        end_idx = end_idx[0] + AP_max
    else:
        end_idx = None
    return start_idx, end_idx


def get_AP_width(v, t, AP_onset, AP_max, AP_end, vrest):
    """
    Computes the width at half maximum of the AP.
    :param AP_onset (int): Index where the membrane potential crosses the AP threshold.
    :param AP_max (int): Index of the maximum of the AP.
    :param AP_end (int): Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace).
    :param vrest (float): Resting potential.
    :return (float): AP width at half maximum.
    """
    start_idx, end_idx = get_AP_width_idxs(v, t, AP_onset, AP_max, AP_end, vrest)
    if start_idx is not None and end_idx is not None:
        return t[end_idx] - t[start_idx]
    else:
        return 0


def get_DAP_amp(v, DAP_max_idx, vrest):
    """
    Computes the amplitude of the DAP in relation to the resting potential.
    :param DAP_max_idx (int): Index of maximum of the DAP.
    :param vrest(float): Resting potential.
    :return (float): Amplitude of the DAP.
    """
    return v[DAP_max_idx] - vrest


def get_DAP_deflection(v, fAHP_min, DAP_max):
    """
    Computes the deflection of the DAP (the height of the depolarization in relation to the minimum of the fAHP).
    :param fAHP_min (int): Index of the Minimum of the fAHP.
    :param DAP_max (int): Index of maximum of the DAP.
    :return (float): Deflection of the DAP.
    """
    return v[DAP_max] - v[fAHP_min]


def get_DAP_width_idx(v, t, fAHP_min, DAP_max, AP_end, vrest):
    """
    Width of the DAP (distance between the time point of the minimum of the fAHP and the time point where the
    downhill side of the DAP is closest to the half amplitude of the minimum of the fAHP).
    :param fAHP_min (int): Index of the Minimum of the fAHP
    :param DAP_max (int): Index of maximum of the DAP.
    :param AP_end (int): Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace)
    :param vrest (float): Resting potential.
    :return (int): Idx where voltage crosses the halfwidth of the DAP.
    """
    halfmax = (v[fAHP_min] - vrest)/2
    halfmax_crossings = np.nonzero(np.diff(np.sign(v[DAP_max:AP_end]-vrest-halfmax)) == -2)[0]
    if len(halfmax_crossings) == 0 or vrest+halfmax > v[fAHP_min]:
        return None
    halfwidth_idx = halfmax_crossings[0] + DAP_max
    return halfwidth_idx

def get_DAP_width(v, t, fAHP_min, DAP_max, AP_end, vrest):
    """
    Width of the DAP (distance between the time point of the minimum of the fAHP and the time point where the
    downhill side of the DAP is closest to the half amplitude of the minimum of the fAHP).
    :param fAHP_min (int): Index of the Minimum of the fAHP
    :param DAP_max (int): Index of maximum of the DAP.
    :param AP_end (int): Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace)
    :param vrest (float): Resting potential.
    :return (float): Width of the DAP.
    """
    halfwidth_idx = get_DAP_width_idx(v, t, fAHP_min, DAP_max, AP_end, vrest)
    if halfwidth_idx is None:
        return 0
    return t[halfwidth_idx] - t[fAHP_min]


def get_v_rest(v, i_inj):
    """
    Computes the resting potential as the mean of the voltage starting at 0 until current is injected.

    :param i_inj (array_like): Injected current (nA).
    :return (float): Resting potential (mean of the voltage trace).
    """
    nonzero = np.nonzero(i_inj)[0]
    if len(nonzero) == 0:
        to_current = -1
    else:
        to_current = nonzero[0]-1
    return np.mean(v[0:to_current])


def get_inputresistance(v, i_inj):
    """Computes the input resistance. Assumes step current protocol: 0 current for some time, step to x current long
    enough to obtain the steady-state voltage.

    :param v (array_like): Voltage (mV) from the step current experiment.
    :param i_inj (array_like): Injected current (nA).
    :return (float): Input resistance (MOhm).
    """
    step = np.nonzero(i_inj)[0]
    idx_step_start = step[0]
    idx_step_half = int(idx_step_start + np.round(len(step)/2.0))
    idx_step_end = step[-1]

    vrest = get_v_rest(v, i_inj)

    vstep = np.mean(v[idx_step_half:idx_step_end])  # start at the middle of the step to get the steady-state voltage

    return (vstep - vrest) / i_inj[idx_step_start]


def get_AP_start_end(v, threshold=-45, n=0):
    AP_onsets = get_AP_onset_idxs(v, threshold)
    if len(AP_onsets) < n+1:
        return None, None
    else:
        AP_onset = AP_onsets[n]
        if len(AP_onsets) < n+2:
            AP_end = -1
        else:
            AP_end = AP_onsets[n+1]
        return AP_onset, AP_end


def get_spike_characteristics(v, t, return_characteristics, v_rest, AP_threshold=-30, AP_max_idx=None, AP_onset=None,
                              AP_interval=None, AP_width_before_onset=0, fAHP_interval=None, std_idx_times=(None, None),
                              k_splines=None, s_splines=None, order_fAHP_min=None, DAP_interval=None, order_DAP_max=None,
                              min_dist_to_DAP_max=None, round_idxs=False,
                              check=False):
    """
    Computes the spike characteristics defined in return_characteristics.

    :param v (np.array): Membrane Potential (must just contain one spike or set up proper intervals).
    :param t (np.array): Time.
    :param return_characteristics (list[str]): Name of characteristics that shall be returned. Options: AP_amp, AP_width, AP_time,
                                   fAHP_amp, fAHP_min_idx, DAP_amp, DAP_deflection, DAP_width, DAP_time, DAP_time_abs,
                                   :type return_characteristics:
                                   DAP_lin_slope, DAP_exp_slope.
    :param AP_threshold: Threshold for detecting APs.
    :param AP_max_idx: Can be used instead of AP threshold to give the location of the AP directly. If used also need
                       to define AP_onset.
    :param: AP_onset: Just define of AP_max_idx is used otherwise it will be inferred using AP_threshold.
    :param v_rest (float): Resting potential.
    :param AP_interval (float): Maximal time (ms) between crossing AP threshold and AP peak.
    :param fAHP_interval (float): Maximal time (ms) between AP peak and fAHP minimum.
    :param std_idx_times (tuple[float, float]): Time (ms) of the start and end indices for the region in which the std of v shall be estimated.
    :param k_splines (int): Degree of the smoothing spline. Must be <= 5.
    :param s_splines (float): Positive smoothing factor used to choose the number of knots. Number of knots will be increased
        until the smoothing condition is satisfied:

            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

        If None (default), ``s = len(w)`` which should be a good value if ``1/w[i]`` is an estimate of the
        standard deviation of ``y[i]``.
        If 0, spline will interpolate through all data points.
    :param order_fAHP_min (float): Time interval (ms) to consider around the minimum for the comparison.
    :param DAP_interval (float): Maximal time (ms) between the fAHP_min and the DAP peak.
    :param order_DAP_max (float): Time interval (ms) to consider around the maximum for the comparison.
    :param min_dist_to_DAP_max:
    :param check (bool): Whether to print and plot the computed values.
    :return (list[float]): Return characteristics.
    """
    dt = t[1] - t[0]

    characteristics = {k: None for k in return_characteristics}

    if round_idxs:
        characteristics['v_rest'] = v_rest
        characteristics['AP_interval_idx'] = int(round(AP_interval / dt)) if not AP_interval is None else None
        characteristics['AP_width_before_onset_idx'] = int(round(AP_width_before_onset / dt))
        characteristics['fAHP_interval_idx'] = int(round(fAHP_interval / dt)) if not fAHP_interval is None else None
        characteristics['std_idxs'] = [int(round(std_idx_time / dt)) if not std_idx_time is None else None
                                       for std_idx_time in std_idx_times]
        characteristics['k_splines'] = k_splines if not k_splines is None else 3
        characteristics['s_splines'] = s_splines
        characteristics['order_fAHP_min_idx'] = int(round(order_fAHP_min / dt)) if not order_fAHP_min is None else 1
        characteristics['DAP_interval_idx'] = int(round(DAP_interval / dt)) if not DAP_interval is None else None
        characteristics['order_DAP_max_idx'] = int(round(order_DAP_max / dt)) if not order_DAP_max is None else 1
        characteristics['min_dist_to_DAP_max'] = int(round(min_dist_to_DAP_max /
                                                        dt)) if not min_dist_to_DAP_max is None else None
    else:
        characteristics['v_rest'] = v_rest
        characteristics['AP_interval_idx'] = to_idx(AP_interval, dt) if not AP_interval is None else None
        characteristics['AP_width_before_onset_idx'] = to_idx(AP_width_before_onset, dt)
        characteristics['fAHP_interval_idx'] = to_idx(fAHP_interval, dt) if not fAHP_interval is None else None
        characteristics['std_idxs'] = [to_idx(std_idx_time, dt) if not std_idx_time is None else None
                                       for std_idx_time in std_idx_times]
        characteristics['k_splines'] = k_splines if not k_splines is None else 3
        characteristics['s_splines'] = s_splines
        characteristics['order_fAHP_min_idx'] = to_idx(order_fAHP_min, dt) if not order_fAHP_min is None else 1
        characteristics['DAP_interval_idx'] = to_idx(DAP_interval, dt) if not DAP_interval is None else None
        characteristics['order_DAP_max_idx'] = to_idx(order_DAP_max, dt) if not order_DAP_max is None else 1
        characteristics['min_dist_to_DAP_max'] = to_idx(min_dist_to_DAP_max, dt) if not min_dist_to_DAP_max is None else None

    if AP_max_idx is None:
        AP_onset, AP_end = get_AP_start_end(v, AP_threshold)
        if AP_onset is None or AP_end is None:
            print ('No AP found!')
            for k in return_characteristics:
                if characteristics[k] is None:
                    characteristics[k] = 0
            # print(characteristics)
            return [characteristics[k] for k in return_characteristics]
        characteristics['AP_max_idx'] = get_AP_max_idx(v, AP_onset, AP_end, interval=characteristics['AP_interval_idx'])
        if characteristics['AP_max_idx'] is None:
            if check:
                check_measures(v, t, characteristics)
            return [characteristics[k] for k in return_characteristics]
    else:
        characteristics['AP_max_idx'] = AP_max_idx

    characteristics['AP_amp'] = get_AP_amp(v, characteristics['AP_max_idx'], characteristics['v_rest'])
    characteristics['AP_width_idxs'] = get_AP_width_idxs(v, t, AP_onset - characteristics['AP_width_before_onset_idx'],
                                                         characteristics['AP_max_idx'],
                                                         AP_onset + characteristics['AP_interval_idx'],
                                                         characteristics['v_rest'])
    characteristics['AP_width'] = get_AP_width(v, t, AP_onset - characteristics['AP_width_before_onset_idx'],
                                               characteristics['AP_max_idx'],
                                               AP_onset + characteristics['AP_interval_idx'],
                                               characteristics['v_rest'])
    characteristics['AP_time'] = t[characteristics['AP_max_idx']]

    std = np.std(v[characteristics['std_idxs'][0]:characteristics['std_idxs'][1]])
    w = np.ones(len(v)) / std
    characteristics['fAHP_min_idx'] = get_fAHP_min_idx_using_splines(v, t, characteristics['AP_max_idx'], len(t),
                                                                     order=characteristics['order_fAHP_min_idx'],
                                                                     interval=characteristics['fAHP_interval_idx'], w=w,
                                                                     k=characteristics['k_splines'],
                                                                     s=characteristics['s_splines'])
    if characteristics['fAHP_min_idx'] is None:
        if check:
            check_measures(v, t, characteristics)
        return [characteristics[k] for k in return_characteristics]
    characteristics['fAHP_amp'] = v[characteristics['fAHP_min_idx']] - v_rest
    characteristics['DAP_max_idx'] = get_DAP_max_idx_using_splines(v, t, int(characteristics['fAHP_min_idx']), len(t),
                                                                   order=characteristics['order_DAP_max_idx'],
                                                                   interval=characteristics['DAP_interval_idx'],
                                                                   min_dist_to_max=characteristics['min_dist_to_DAP_max'],
                                                                   w=w,
                                                                   k=characteristics['k_splines'],
                                                                   s=characteristics['s_splines'])
    if characteristics['DAP_max_idx'] is None:
        if check:
            check_measures(v, t, characteristics)
        return [characteristics[k] for k in return_characteristics]

    characteristics['DAP_amp'] = get_DAP_amp(v, int(characteristics['DAP_max_idx']), characteristics['v_rest'])
    characteristics['DAP_deflection'] = get_DAP_deflection(v, int(characteristics['fAHP_min_idx']),
                                                           int(characteristics['DAP_max_idx']))
    characteristics['DAP_width_idx'] = get_DAP_width_idx(v, t, int(characteristics['fAHP_min_idx']),
                                                         int(characteristics['DAP_max_idx']), len(t),
                                                         characteristics['v_rest'])
    characteristics['DAP_width'] = get_DAP_width(v, t, int(characteristics['fAHP_min_idx']),
                                                 int(characteristics['DAP_max_idx']), len(t), characteristics['v_rest'])
    characteristics['DAP_time'] = t[int(round(characteristics['DAP_max_idx']))] - t[characteristics['AP_max_idx']]
    characteristics['DAP_time_abs'] = t[int(round(characteristics['DAP_max_idx']))]
    characteristics['fAHP2DAP_time'] = t[int(round(characteristics['DAP_max_idx']))] - t[characteristics['fAHP_min_idx']]

    if characteristics['DAP_width_idx'] is None:
        if check:
            check_measures(v, t, characteristics)
        return [characteristics[k] for k in return_characteristics]

    characteristics['half_fAHP_crossings'] = np.nonzero(np.diff(np.sign(v[int(characteristics['DAP_max_idx']):len(t)]
                                                     - v[int(characteristics['fAHP_min_idx'])])) == -2)[0]
    if len(characteristics['half_fAHP_crossings']) == 0:
        if check:
            check_measures(v, t, characteristics)
        return [characteristics[k] for k in return_characteristics]

    half_fAHP_idx = characteristics['half_fAHP_crossings'][0] + characteristics['DAP_max_idx']
    characteristics['slope_start'] = half_fAHP_idx
    characteristics['slope_end'] = len(t) - 1
    characteristics['DAP_lin_slope'] = np.abs((v[int(characteristics['slope_end'])] - v[int(characteristics['slope_start'])])
                              / (t[int(characteristics['slope_end'])] - t[int(characteristics['slope_start'])]))

    try:
        characteristics['DAP_exp_slope'] = curve_fit(
            partial(exp_fit, v=v[int(characteristics['slope_start']):int(characteristics['slope_end'])]),
            np.arange(characteristics['slope_end']-int(characteristics['slope_start'])) * dt,
            v[int(characteristics['slope_start']):int(characteristics['slope_end'])],
            p0=1, bounds=(0, np.inf))[0][0]
    except RuntimeError:
        characteristics['DAP_exp_slope'] = None
    if check:
        check_measures(v, t, characteristics)
        for k in return_characteristics:
            if characteristics[k] is None:
                characteristics[k] = 0
    return [characteristics[k] for k in return_characteristics]


def check_measures(v, t, characteristics):
    dt = t[1] - t[0]
    if not characteristics.get('AP_max_idx') is None:
        print ('AP_amp (mV): ', characteristics['AP_amp'])
        print ('AP_width (ms): ', characteristics['AP_width'])
        if not characteristics.get('fAHP_min_idx') is None:
            print ('fAHP_amp: (mV): ', characteristics['fAHP_amp'])
        if not characteristics.get('DAP_max_idx') is None:
            print ('DAP_amp: (mV): ', characteristics['DAP_amp'])
            print ('DAP_deflection: (mV): ', characteristics['DAP_deflection'])
            print ('DAP_width: (ms): ', characteristics['DAP_width'])
            print ('DAP time: (ms): ', characteristics['DAP_time'])
            if not characteristics.get('DAP_exp_slope') is None:
                print ('DAP_exp_slope: ', characteristics['DAP_exp_slope'])
                print ('DAP_lin_slope: ', characteristics['DAP_lin_slope'])

    pl.figure()
    pl.plot(t, v)
    pl.plot(t[characteristics['AP_max_idx']], v[characteristics['AP_max_idx']], 'or', label='AP_max_idx')
    if not characteristics.get('AP_width_idxs')[0] is None and not characteristics.get('AP_width_idxs')[1] is None:
        pl.plot(t[np.array(characteristics['AP_width_idxs'])],
                v[np.array(characteristics['AP_width_idxs'])], '-or', label='AP_width')
    if not characteristics.get('fAHP_min_idx') is None:
        pl.plot(t[int(characteristics['fAHP_min_idx'])], v[int(characteristics['fAHP_min_idx'])], 'og', label='fAHP')
        if not characteristics.get('DAP_max_idx') is None:
            pl.plot(t[int(characteristics['DAP_max_idx'])], v[int(characteristics['DAP_max_idx'])], 'ob',
                    label='DAP_max')
        if not characteristics.get('DAP_width_idx') is None:
            pl.plot([t[int(characteristics['fAHP_min_idx'])], t[int(characteristics['DAP_width_idx'])]],
                    [v[int(characteristics['fAHP_min_idx'])]
                     - (v[int(characteristics['fAHP_min_idx'])] - characteristics['v_rest']) / 2,
                     v[int(characteristics['DAP_width_idx'])]],
                    '-ob', label='DAP_width')
        if not characteristics.get('slope_start') is None and not characteristics.get('slope_end') is None:
            pl.plot([t[int(characteristics['slope_start'])], t[int(characteristics['slope_end'])]],
                    [v[int(characteristics['slope_start'])], v[int(characteristics['slope_end'])]],
                    '-oy', label='lin_slope')

            pl.plot(t[int(characteristics['slope_start']): int(characteristics['slope_end'])],
                    exp_fit(np.arange(int(characteristics['slope_end'])-int(characteristics['slope_start'])) * dt,
                            characteristics['DAP_exp_slope'],
                            v[int(characteristics['slope_start']):int(characteristics['slope_end'])]), 'y',
                    label='exp_slope')
    pl.legend()
    pl.show()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as pl

    # # test on experimental data
    data_dir = '../data/2015_08_26b/raw/rampIV/3.0(nA).csv'
    data = pd.read_csv(data_dir)
    v_exp = np.array(data.v)
    i_exp = np.array(data.i)
    t_exp = np.array(data.t)
    dt_exp = t_exp[1]-t_exp[0]

    vrest = get_v_rest(v_exp, i_exp)
    AP_onsets = get_AP_onset_idxs(v_exp, threshold=-30)
    AP_onset = AP_onsets[0]
    AP_end = -1

    AP_max = get_AP_max_idx(v_exp, AP_onset, AP_end, interval=1 / dt_exp)
    fAHP_min = get_fAHP_min_idx(v_exp, AP_max, AP_end, interval=5 / dt_exp)
    DAP_max = get_DAP_max_idx(v_exp, fAHP_min, AP_end, interval=10 / dt_exp)

    AP_amp = get_AP_amp(v_exp, AP_max, vrest)
    AP_width = get_AP_width(v_exp, t_exp, AP_onset, AP_max, AP_end, vrest)
    DAP_amp = get_DAP_amp(v_exp, DAP_max, vrest)
    DAP_deflection = get_DAP_deflection(v_exp, DAP_max, fAHP_min)
    DAP_width = get_DAP_width(v_exp, t_exp, fAHP_min, DAP_max, AP_end, vrest)
    print ('AP amplitude: ' + str(AP_amp) + ' (mV)')
    print ('AP width: ' + str(AP_width) + ' (ms)')
    print ('DAP amplitude: ' + str(DAP_amp) + ' (mV)')
    print ('DAP deflection: ' + str(DAP_deflection) + ' (mV)')
    print ('DAP width: ' + str(DAP_width) + ' (ms)')

    pl.figure()
    pl.plot(t_exp, v_exp, 'k', label='V')
    pl.plot(t_exp[AP_onsets], v_exp[AP_onsets], 'or', label='AP onsets')
    pl.plot(t_exp[AP_max], v_exp[AP_max], 'ob', label='AP maximum')
    pl.plot(t_exp[fAHP_min], v_exp[fAHP_min], 'oy', label='fAHP minimum')
    pl.plot(t_exp[DAP_max], v_exp[DAP_max], 'og', label='DAP maximum')
    pl.legend()
    pl.show()

    data_dir = '../data/2015_08_26b/raw/rampIV/3.0(nA).csv'
    data = pd.read_csv(data_dir)
    v_step = np.array(data.v)
    i_step = np.array(data.i)
    t_step = np.array(data.t)
    dt_step = t_step[1]-t_step[0]

    rin = get_inputresistance(v_step, i_step)
    print ('Input resistance: ' + str(rin) + ' (MOhm)')
