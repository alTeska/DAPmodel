import numpy as np
from cell_fitting.read_heka import get_sweep_index_for_amp, get_i_inj_from_function, get_v_and_t_from_heka, shift_v_rest
from DAPmodel.analyze_APs import get_spike_characteristics,  get_spike_characteristics_dict

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats


class DAPSummaryStats(BaseSummaryStats):
    """SummaryStats class for the DAP model

    Calculates summary statistics of AP/DAP:
    AP_amp, AP_width, DAP_amp, DAP_width, DAP_deflection, DAP_time
    """
    def __init__(self, t_on, t_off, n_summary=6, seed=None):
        """See SummaryStats.py for docstring"""
        super(DAPSummaryStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off
        self.n_summary = n_summary

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

            return_characteristics = ['AP_amp', 'AP_width', 'DAP_amp', 'DAP_width',
                          'DAP_deflection', 'DAP_time']
            spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)  # standard parameters to use

            characteristics = get_spike_characteristics(v, t, return_characteristics,
                                                        v_rest=v[0], std_idx_times=(0, 1)
                                                        ,**spike_characteristics_dict)

            rest_pot = np.mean(x['data'][t<t_on])

            sum_stats_vec = np.array([
                            rest_pot,
                            characteristics['AP_amp'],
                            characteristics['AP_width'],
                            characteristics['DAP_amp'],
                            characteristics['DAP_width'],
                            characteristics['DAP_deflection'],
                            characteristics['DAP_time']
                            ])

            sum_stats_vec = sum_stats_vec[0:self.n_summary]
            stats.append(sum_stats_vec)

        return np.asarray(stats)
