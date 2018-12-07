import abc
import numpy as np

class DAPBaseC(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, state, params, seed=None, **kwargs):
        self.state = np.asarray(state)
        self.params = np.asarray(params)


    # model integration
    def x_inf(self, V, x_vh, x_vs):
        '''steady state values'''
        return 1 / (1 + np.exp((x_vh - V) / x_vs))

    def x_tau(self, V, xinf, ion_ch):
        return (ion_ch['tau_min'] + (ion_ch['tau_max'] - ion_ch['tau_min']) * \
                xinf * np.exp(ion_ch['tau_delta'] * \
                (ion_ch['vh'] - V) / ion_ch['vs']))
