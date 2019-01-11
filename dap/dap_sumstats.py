import numpy as np
from scipy.signal import argrelmin, argrelmax

from .cell_fitting.read_heka import get_sweep_index_for_amp, get_i_inj_from_function, get_v_and_t_from_heka, shift_v_rest
from dap.analyze_APs import get_spike_characteristics,  get_spike_characteristics_dict

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats


class DAPSummaryStats(BaseSummaryStats):
    """SummaryStats class for the DAP model

    Calculates summary statistics of AP/DAP:
    AP_amp, AP_width, DAP_amp, DAP_width, DAP_deflection, DAP_time

    Version excluding the analyze_APs file, recalculates the values
    """
    def __init__(self, t_on, t_off, n_summary=6, seed=None):
        """See SummaryStats.py for docstring"""
        super(DAPSummaryStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off
        self.n_summary = n_summary

        data_dir = '/home/ateska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell
        protocol = 'rampIV' # 'IV' # 'rampIV' # 'Zap20'
        ramp_amp = 3.1 # steps of 0.05 -0.15
        v_shift = -16  # shift for accounting for the liquid junction potential

        if protocol == 'Zap20':
            sweep_idx = 0
        else:
            sweep_idx = get_sweep_index_for_amp(ramp_amp, protocol)

        self.v0, self.t = get_v_and_t_from_heka(data_dir, protocol, sweep_idxs=[sweep_idx])
        self.v0 = shift_v_rest(self.v0[0], v_shift)


    def calc(self, repetition_list):
        """Calculate summary statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
        stats = []
        stats_idx = []
        for r in range(len(repetition_list)):
            x = repetition_list[r]

            N = x['data'].shape[0]
            t = x['time']
            dt = x['dt']
            t_on = self.t_on
            t_off = self.t_off

            # initialise array of spike counts
            v = np.array(x['data'])

            # resting potential
            rest_pot = np.mean(v[t<t_on])
            rest_pot_std = np.std(v[int(.9*t_on/dt):int(t_on/dt)])   # TODO: add if needed

            # RMSE
            n = len(self.v0)


            # more then one AP:
            multiple_AP = np.shape(np.where(v > 30))[1]

            #case without any action potential or more then one AP
            if (np.all(v <= 20)) or (multiple_AP > 100):
                AP_onsets = 999
                AP_amp = 999
                AP_width = 999
                DAP_amp = 999
                DAP_width = 999
                DAP_deflection = 999
                DAP_time = 999
                mAHP = 999

            else:
                threshold = -30
                # hyperpolarization after DAP
                mAHP_idx = np.argmin(v)
                mAHP = v[mAHP_idx]

                # Action potential
                AP_onsets = np.where(v > threshold)[0]
                AP_start = AP_onsets[0]
                AP_end = AP_onsets[-1]
                AP_max_idx = AP_start + np.argmax(v[AP_start:AP_end])
                AP_max = v[AP_max_idx]
                AP_amp = AP_max - rest_pot

                # AP width
                AP_onsets_half_max = np.where(v > (AP_max+rest_pot)/2)[0]
                AP_width = t[AP_onsets_half_max[-1]] - t[AP_onsets_half_max[0]]

                # if np.size(AP_onsets_half_max) > 1:
                #     AP_width = t[AP_onsets_half_max[-1]] - t[AP_onsets_half_max[0]]
                # else:
                #     AP_width = 999

                # DAP: fAHP
                v_dap = v[AP_max_idx:]
                fAHP_idx = argrelmin(v[AP_max_idx:])[0][0] + AP_max_idx
                fAHP = v[fAHP_idx]

                # DAP amplitude
                DAP_max_idx = argrelmax(v_dap)[0][1] + AP_max_idx
                DAP_max = v[DAP_max_idx]
                DAP_amp = DAP_max - rest_pot

                DAP_deflection = DAP_amp - (fAHP - rest_pot)
                DAP_time = t[DAP_max_idx] - t[AP_max_idx]    # Time between AP and DAP maximum

                # Width of DAP: between fAHP and halfsize of fAHP after DAP max
                vnorm = v[DAP_max_idx:] - rest_pot

                if np.any((abs(vnorm) < abs(fAHP - rest_pot)/2)):
                    half_max = np.where((abs(vnorm) < abs(fAHP - rest_pot)/2))[0]

                    DAP_width_idx = DAP_max_idx + half_max[0]
                    DAP_width = (DAP_width_idx - fAHP_idx) * dt
                else:
                    DAP_width = 999



            print('rest_pot', rest_pot)
            print('AP_amp', AP_amp)
            print('AP_width', AP_width)
            print('fAHP', fAHP)
            print('DAP_amp', DAP_amp)
            print('DAP_width', DAP_width)
            print('DAP_deflection', DAP_deflection)
            print('DAP_time', DAP_time)
            print('mAHP', mAHP)

            sum_stats_vec = np.array([
                            # rmse,
                            rest_pot,
                            AP_amp,
                            AP_width,
                            fAHP,
                            DAP_amp,
                            DAP_width,
                            DAP_deflection,
                            DAP_time,
                            mAHP,
                            ])

            sum_stats_vec_inx = np.array([
                            AP_max_idx,
                            fAHP_idx,
                            mAHP_idx,
                            DAP_max_idx,
                            DAP_width_idx,
                            ])


            sum_stats_vec = sum_stats_vec[0:self.n_summary]
            stats.append(sum_stats_vec)
            stats_idx.append(sum_stats_vec_inx)
            # print('summary statistics', sum_stats_vec)

        return np.asarray(stats), np.asarray(stats_idx)
