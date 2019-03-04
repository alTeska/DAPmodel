import numpy as np
from scipy import stats as spstats
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats


class DAPSummaryStatsStepMoments(BaseSummaryStats):
    """SummaryStats class for the DAP model

    Calculates summary statistics of AP/DAP with step current function:
    Summary statistics based on statistical moments of the signal. Features
    here consist of moments calculated with spstats function of given order
    (means, std, skewness etc.) and autocorrelation of the signal.

    Moments are calculated during the step current input. Inter spikes intervals
    are also used as features.
    """
    def __init__(self, t_on, t_off,n_mom=5, n_xcorr=4, n_summary=2, seed=None):

        """See SummaryStats.py for docstring"""
        super(DAPSummaryStatsStepMoments, self).__init__(seed=seed)
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

            # spike times
            spike_times = np.array(t)[ind[0]]
            spike_times_stim = spike_times[(spike_times > t_on) & (spike_times < t_off)]

            # number of spikes
            if spike_times_stim.shape[0] > 0:
                spike_times_stim = spike_times_stim[np.append(1, np.diff(spike_times_stim))>0.5]

            spikes = len(spike_times_stim)

            # ISI TODO: decide if you want to add higher moments
            ISI = np.diff(spike_times_stim).astype(float)
            ind = [0,1,-1]
            ISI1 = np.array([1000.]*3)
            ISI1[0:np.maximum(0,spike_times_stim.shape[0]-1)] = ISI[ind[0:np.maximum(0,spike_times_stim.shape[0]-1)]]

            if spike_times_stim.shape[0] > 1:
                ISImom = np.array([np.mean(ISI), np.std(ISI)])
            else:
                ISImom = np.array([t_off, 0.])

            # accommodation index
            if spike_times_stim.shape[0] < 3:
               A_ind = 1000
            else:
               ISI = np.diff(spike_times_stim)
               A_ind = np.mean( [ (ISI[i_min+1]-ISI[i_min])/(ISI[i_min+1]+ISI[i_min]) for i_min in range (0,ISI.shape[0]-1)] )

            # auto-correlations
            x_on_off = x['data'][(t > t_on) & (t < t_off)]-np.mean(x['data'][(t > t_on) & (t < t_off)])
            x_corr_val = np.dot(x_on_off, x_on_off)

            xcorr_steps = np.linspace(1./dt, self.n_xcorr*1./dt, self.n_xcorr).astype(int)
            x_corr_full = np.zeros(self.n_xcorr)

            for ii in range(self.n_xcorr):
                x_on_off_part = np.concatenate((x_on_off[xcorr_steps[ii]:], np.zeros(xcorr_steps[ii])))
                x_corr_full[ii] = np.dot(x_on_off,x_on_off_part)

            x_corr1 = x_corr_full/x_corr_val

            # moments of the signal
            std_pw = np.power(np.std(x['data'][(t > t_on) & (t < t_off)]), np.linspace(3,self.n_mom,self.n_mom-2))
            std_pw = np.concatenate((np.ones(1),std_pw))
            moments = spstats.moment(x['data'][(t > t_on) & (t < t_off)], np.linspace(2,self.n_mom,self.n_mom-1))/std_pw

            print('spikes:', spikes)

            sum_stats_vec = np.concatenate((
                    np.array([rest_pot,rest_pot_std,np.mean(x['data'][(t > t_on) & (t < t_off)]), spikes]),
                    np.array([spike_times_stim.shape[0]]),
                    x_corr1,
                    moments,
                    ISI1,
                ))

            sum_stats_vec = sum_stats_vec[0:self.n_summary]

            stats.append(sum_stats_vec)

        return np.asarray(stats)
