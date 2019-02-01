import numpy as np
from scipy import stats as spstats
# from scipy.signal import argrelmin, argrelmax

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats

class DAPSummaryStatsMoments(BaseSummaryStats):
    """SummaryStats class for the DAP model

    Calculates summary statistics of AP/DAP:
    AP_amp, AP_width, DAP_amp, DAP_width, DAP_deflection, DAP_time

    Version excluding the analyze_APs file, recalculates the values
    """
    def __init__(self, t_on, t_off,n_mom = 5, n_xcorr = 4, n_summary=2, seed=None):
        """See SummaryStats.py for docstring"""
        super(DAPSummaryStatsMoments, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off
        self.n_summary = n_summary
        self.n_xcorr = n_xcorr
        self.n_mom = n_mom


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

            # resting potential and std
            rest_pot = np.mean(x['data'][t<t_on])
            rest_pot_std = np.std(x['data'][int(.9*t_on/dt):int(t_on/dt)])

            # initialise array of spike counts
            v = np.array(x['data'])

            # put everything to -10 that is below -10 or has negative slope
            ind = np.where(v < -10)
            v[ind] = -10
            ind = np.where(np.diff(v) < 0)
            v[ind] = -10

            # remaining negative slopes are at spike peaks
            ind = np.where(np.diff(v) < 0)
            spike_times = np.array(t)[ind]
            spikes = np.shape(np.array(t)[ind])[0]

            # find relevant lenght of the trace (for now from t_on till the end)
            v_dap = np.array(x['data'])
            v_all = v_dap[(t > t_on) & (t < 100)]

            # TODO: optionally cut the tail as well but might not work with
            # bad trace: use v2 instead of dap for further calcualtions -> 100 ms for now
            # ind = np.where(v_dap > rest_pot)
            # v2 = v_dap[ind[0][0]:ind[0][-1]]

            # normalize for autokorrelations
            x_on_off = v_all - np.mean(v_all)
            x_corr_val = np.dot(x_on_off,x_on_off)
            xcorr_steps = np.linspace(1./dt,self.n_xcorr*1./dt,self.n_xcorr).astype(int)
            x_corr_full = np.zeros(self.n_xcorr)
            for ii in range(self.n_xcorr):
                x_on_off_part = np.concatenate((x_on_off[xcorr_steps[ii]:],np.zeros(xcorr_steps[ii])))
                x_corr_full[ii] = np.dot(x_on_off,x_on_off_part)

            x_corr1 = x_corr_full/x_corr_val

            # moments of the signal
            v_AP  = v_dap[(t >= t_on) & (t <= t_off+1)]
            v_DAP = v_dap[(t > t_off+1) & (t < 100)]

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1,3)
            # ax[0].plot(v_all)
            # ax[1].plot(v_AP)
            # ax[2].plot(v_DAP)
            # plt.show(   )

            std_pw_AP = np.power(np.std(v_AP), np.linspace(3,self.n_mom,self.n_mom-2))
            std_pw_AP = np.concatenate((np.ones(1),std_pw_AP))
            moments_AP = spstats.moment(v_AP, np.linspace(2,self.n_mom,self.n_mom-1))/std_pw_AP

            std_pw_DAP = np.power(np.std(v_DAP), np.linspace(3,self.n_mom,self.n_mom-2))
            std_pw_DAP = np.concatenate((np.ones(1),std_pw_DAP))
            moments_DAP = spstats.moment(v_DAP, np.linspace(2,self.n_mom,self.n_mom-1))/std_pw_DAP

            # concatenation of summary statistics
            # try:
            sum_stats_vec = np.concatenate((
                    np.array([rest_pot,rest_pot_std, np.mean(v_dap), spikes]),
                    x_corr1,
                    moments_AP,
                    moments_DAP,
                    # spikes,
                ))

            sum_stats_vec = sum_stats_vec[0:self.n_summary]
            # except:
                # return None

            stats.append(sum_stats_vec)
            # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            # print('s', sum_stats_vec)

        return np.asarray(stats)
