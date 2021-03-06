import os
import re
from dap.cell_fitting.custom_hekareader import HekaReader
from dap.cell_fitting.i_inj_functions import *
from dap.cell_fitting.util import init_nan, convert_to_unit

__author__ = "Caroline Fischer"


def load_data(data_dir, protocol, amp, v_shift=-16):
    if protocol == 'Zap20':
        sweep_idxs = [0]
        print ('amp not used!')
    else:
        sweep_idxs = [get_sweep_index_for_amp(amp, protocol)]
    v, t = get_v_and_t_from_heka(data_dir, protocol, sweep_idxs=sweep_idxs)
    v = shift_v_rest(v[0], v_shift)
    t = t[0]
    i_inj = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])[0]
    return v, t, i_inj


def get_v_and_t_from_heka(file_dir, protocol, group='Group1', trace='Trace1', sweep_idxs=None, return_series=False,
                          return_sweep_idxs=False, heka_dict=None):
    if heka_dict is None:
        hekareader = HekaReader(file_dir)
        type_to_index = hekareader.get_type_to_index()
        protocol_to_series = hekareader.get_protocol(group)
    else:
        hekareader = heka_dict['hekareader']
        type_to_index = heka_dict['type_to_index']
        protocol_to_series = heka_dict['protocol_to_series']
    series = protocol_to_series[protocol]
    sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series]) + 1)]
    # print '# sweeps: ', len(sweeps)
    if sweep_idxs is None:
        sweep_idxs = range(len(sweeps))
    sweeps = [sweeps[index] for index in sweep_idxs]
    indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

    v = [0] * len(indices)
    t = [0] * len(indices)
    for i, index in enumerate(indices):
        t[i], v[i] = hekareader.get_xy(index)
        x_unit, y_unit = hekareader.get_units_xy(index)
        assert x_unit == 's'
        assert y_unit == 'V'
        t[i] = convert_to_unit('m', t[i])  # ms
        v[i] = convert_to_unit('m', v[i])  # mV
        t[i] = t[i].tolist()
        v[i] = v[i].tolist()

    return_idxs = np.array([True, True, return_series, return_sweep_idxs])
    return np.array([np.array(v), np.array(t), series, sweep_idxs])[return_idxs]  # not matrix if v[i]s have different length


def get_i_inj_from_function(protocol, sweep_idxs, tstop, dt, return_discontinuities=False):
    i_inj = init_nan((len(sweep_idxs), to_idx(tstop, dt)+1))
    discontinuities = []
    for i, sweep_idx in enumerate(sweep_idxs):
        if protocol == 'IV':
            step_amp = np.round(-0.15 + sweep_idx * 0.05, 2)
            start_step = 250  # ms
            end_step = 750  # ms
            discontinuities = [start_step, end_step]
            i_inj[i] = get_i_inj_step(start_step, end_step, step_amp, tstop, dt)
        elif protocol == 'hypTester':
            step_amp = -0.005
            start_step = 200  # ms
            end_step = 600  # ms
            discontinuities = [start_step, end_step]
            i_inj[i] = get_i_inj_step(start_step, end_step, step_amp, tstop, dt)
        elif protocol == 'rampIV':
            ramp_amp = np.round(0.1 + sweep_idx * 0.1, 2)
            ramp_start = 10.0  # ms
            ramp_peak = 10.8  # ms
            ramp_end = 12.0  # ms
            amp_before = 0
            amp_after = 0
            discontinuities = [ramp_start, ramp_peak, ramp_end]
            i_inj[i] = get_i_inj_rampIV(ramp_start, ramp_peak, ramp_end, amp_before, ramp_amp, amp_after, tstop, dt)
        elif protocol == 'Zap20':
            discontinuities = np.arange(0, tstop+dt, dt)
            i_inj[i] = get_i_inj_zap(amp=0.1, freq0=0, freq1=20, onset_dur=2000, offset_dur=2000 - dt, zap_dur=30000,
                                     tstop=tstop, dt=dt)
        elif 'hyperRampTester' in protocol:
            p_idx = int(protocol[-2:-1])
            step_amp = -0.05 + p_idx * -0.05
            print (step_amp)
            step_start = 200
            step_end = 600
            ramp_len = 2
            discontinuities = [step_start, step_end, step_end+ramp_len]
            i_inj[i] = get_i_inj_hyper_depo_ramp(step_start, step_end, ramp_len, step_amp, tstop=tstop, dt=dt)
        elif 'depoRampTester' in protocol:
            p_idx = int(protocol[-2:-1])
            step_amp = 0.05 + p_idx * 0.05
            step_start = 200
            step_end = 600
            ramp_len = 2
            discontinuities = [step_start, step_end, step_end+ramp_len]
            i_inj[i] = get_i_inj_hyper_depo_ramp(step_start, step_end, ramp_len, step_amp, tstop=tstop, dt=dt)
        # elif protocol == 'PP':
        #     ramp_amp = 4.0
        #     ramp3_amp = 1.0
        #     ramp3_time = get_ramp3_times()[0]
        #     step_amp = 0
        #     len_step = 250
        #     len_ramp = 2
        #     start_ramp1 = 20
        #     i_inj[i] = get_i_inj_double_ramp(ramp_amp, ramp3_amp, ramp3_time, step_amp, len_step, len_ramp=len_ramp, start_ramp1=start_ramp1, tstop=tstop, dt=dt)
        else:
            raise ValueError('No function saved for this protocol!')
    if return_discontinuities:
        return i_inj, discontinuities
    if protocol == 'IV':
        return i_inj, start_step, end_step
    elif protocol == 'rampIV':
        return i_inj, ramp_start, ramp_end




def get_i_inj_standard_params(protocol, sweep_idxs=None):
    if protocol == 'IV':
        params = {
            'step_amp': [np.round(-0.15 + sweep_idx * 0.05, 2) for sweep_idx in sweep_idxs],  # nA
            'start_step': 250,  # ms
            'end_step': 750,  # ms
            'tstop': 1149.95,  # ms
            'dt': 0.05  # ms
        }
    elif protocol == 'hypTester':
        params = {
            'step_amp': -0.005,  # nA
            'start_step': 200,  # ms
            'end_step': 600,  # ms
            'tstop': 999.95,  # ms
            'dt': 0.05  # ms
        }
    elif protocol == 'rampIV':
        params = {
            'ramp_amp': [np.round(0.1 + sweep_idx * 0.1, 2) for sweep_idx in sweep_idxs],  # nA
            'ramp_start': 10.0,  # ms
            'ramp_peak': 10.8,  # ms
            'ramp_end': 12.0,  # ms
            'amp_before': 0.0,  # nA
            'amp_after': 0.0,  # nA
            'tstop': 161.99,  # ms
            'dt': 0.01  # ms
        }
    elif protocol == 'Zap20':
        params = {
            'amp': 0.1,
            'freq0': 0,
            'freq1': 20,
            'onset_dur': 2000,
            'offset_dur': 2000 - 0.025,
            'zap_dur': 30000,
            'tstop': 33999.975,
            'dt': 0.025
        }
    elif 'hyperRampTester' in protocol:
        p_idx = int(protocol[-2:-1])
        params = {
            'step_amp': -0.05 + p_idx * -0.05,
            'step_start': 200,
            'step_end': 600,
            'ramp_len': 2,
            'tstop': 1001.95,
            'dt': 0.05
        }
    elif 'depoRampTester' in protocol:
        p_idx = int(protocol[-2:-1])
        params = {
            'step_amp': 0.05 + p_idx * 0.05,
            'step_start': 200,
            'step_end': 600,
            'ramp_len': 2,
            'tstop': 1001.95,
            'dt': 0.05
        }
    elif protocol == 'PP':
        params = {
            'ramp_amp': 4.0,  # nA
            'ramp3_amp': 1.0,  # nA
            'ramp3_time': get_ramp3_times()[0],  # ms
            'step_amp': 0,  # nA
            'len_step': 250,  # ms
            'len_ramp': 2,  # ms
            'start_ramp1': 20,  # ms
            'tstop': 691.99,  # ms
            'dt': 0.01  # ms
        }
    else:
        raise ValueError('No function saved for this protocol!')
    return params


def get_sweep_index_for_amp(amp, protocol):
    if protocol == 'IV':
        sweep_idx = np.round((amp + 0.15) / 0.05, 10)  # rounding necessary for integer recognition and conversion
        assert sweep_idx.is_integer()
        sweep_idx = int(sweep_idx)
    elif protocol == 'rampIV':
        sweep_idx = np.round((amp - 0.1) / 0.1, 10)
        assert sweep_idx.is_integer()
        sweep_idx = int(sweep_idx)
    else:
        raise ValueError('Conversion not available for this protocol!')
    return sweep_idx


def set_v_rest(v, v_rest_old, vrest_new):
    return v - (v_rest_old - vrest_new)


def shift_v_rest(v, v_rest_shift):
    return v + v_rest_shift
