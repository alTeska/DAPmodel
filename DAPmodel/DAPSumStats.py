import numpy as np
from scipy.signal import argrelmin, argrelmax

from .cell_fitting.read_heka import get_sweep_index_for_amp, get_i_inj_from_function, get_v_and_t_from_heka, shift_v_rest
from DAPmodel.analyze_APs import get_spike_characteristics,  get_spike_characteristics_dict

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats


class DAPSummaryStatsA(BaseSummaryStats):
    """SummaryStats class for the DAP model

    Calculates summary statistics of AP/DAP:
    AP_amp, AP_width, DAP_amp, DAP_width, DAP_deflection, DAP_time

    Version excluding the analyze_APs file, recalculates the values
    """
    def __init__(self, t_on, t_off, n_summary=6, seed=None):
        """See SummaryStats.py for docstring"""
        super(DAPSummaryStatsA, self).__init__(seed=seed)
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
            rest_pot = np.mean(x['data'][t<t_on])
            rest_pot_std = np.std(x['data'][int(.9*t_on/dt):int(t_on/dt)])   # TODO: add if needed

            # RMSE
            n = len(self.v0)
            rmse = np.linalg.norm(v - self.v0) / np.sqrt(n)

            # Action potential
            threshold = -30
            if np.any(v > threshold):
                AP_onsets = np.where(v > threshold)[0]
                AP_start = AP_onsets[0]
                AP_end = AP_onsets[-1]
                AP_max_idx = AP_start + np.argmax(v[AP_start:AP_end])
                AP_max = v[AP_max_idx]
                AP_amp = AP_max - rest_pot

                # AP width
                AP_onsets_half_max = np.where(v > AP_max+rest_pot/2)[0]
                if np.size(AP_onsets_half_max) > 1:
                    AP_width = t[AP_onsets_half_max[-1]] - t[AP_onsets_half_max[0]]
                else:
                    AP_width = 0

                # DAP: fAHP
                fAHP_idx = argrelmin(v)[0][1]
                fAHP = v[fAHP_idx] - rest_pot

                # DAP amplitude
                DAP_max_idx = argrelmax(v)[0][1]
                DAP_max = v[DAP_max_idx]
                DAP_amp = DAP_max - rest_pot

                DAP_deflection = DAP_amp - fAHP
                DAP_time = t[DAP_max_idx] - t[AP_max_idx]    # Time between AP and DAP maximum

                # Width of DAP: between fAHP and halfsize of fAHP after DAP max
                vnorm = v[fAHP_idx:-1] - rest_pot

                if np.any(vnorm < fAHP/2):
                    half_max = np.where(vnorm < fAHP/2)[0]
                    DAP_width = half_max[0] * dt
                else:
                    DAP_width = 0

            else:
                #case without any action potential
                AP_onsets = 0
                AP_amp = 0
                AP_width = 0
                DAP_amp = 0
                DAP_width = 0
                DAP_deflection = 0
                DAP_time = 0


            sum_stats_vec = np.array([
                            rest_pot,
                            rmse,
                            AP_amp,
                            AP_width,
                            DAP_amp,
                            DAP_width,
                            DAP_deflection,
                            DAP_time,
                            # rest_pot_std,  # TODO: decide about keeping it
                            ])

            sum_stats_vec = sum_stats_vec[0:self.n_summary]
            stats.append(sum_stats_vec)
            # print('summary statistics', sum_stats_vec)

        return np.asarray(stats)
