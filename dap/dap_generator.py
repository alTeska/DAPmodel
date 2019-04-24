import delfi.distribution as dd
import numpy as np

from delfi.generator.BaseGenerator import BaseGenerator



class DAPDefault(BaseGenerator):
    @copy_ancestor_docstring
    def _feedback_proposed_param(self, param):
        # See BaseGenerator for docstring

        # if prior is uniform, reject samples outside of bounds
        # samples might be drawn outside bounds due to proposal
        if isinstance(self.prior, dd.Uniform):
            if np.any(param < self.prior.lower) or \
               np.any(param > self.prior.upper):
                return 'resample'
        elif isinstance(self.prior, dd.IndependentJoint):
            for j, p in enumerate(self.prior.dists):
                ii = self.prior.dist_index_eachdim == j
                if isinstance(p, dd.Uniform):
                    if np.any(param[:, ii] < p.lower) or \
                       np.any(param[:, ii] > p.upper):
                        return 'resample'

                elif isinstance(p, dd.Gamma):
                    if np.any(param[:,ii] < p.offset):
                        return 'resample'

        return 'accept'

    @copy_ancestor_docstring
    def _feedback_forward_model(self, data):
        # See BaseGenerator for docstring
        return 'accept'

    @copy_ancestor_docstring
    def _feedback_summary_stats(self, sum_stats):
        # See BaseGenerator for docstring
        return 'accept'

    def process_batch(self, params_batch, result, skip_feedback=False):
        flag = True
        ret_stats = []
        ret_params = []

        # for every datum in data, check validity
        params_data_valid = []  # list of params with valid data
        data_valid = []  # list of lists containing n_reps dicts with data

        for param, datum in zip(params_batch, result):
            # check validity
            response = self._feedback_forward_model(datum)
            if response == 'accept' or skip_feedback:
                data_valid.append(datum)
                # if data is accepted, accept the param as well
                params_data_valid.append(param)
            elif response == 'discard':
                continue
            else:
                raise ValueError('response not supported')

        # for every data in data, calculate summary stats
        for param, datum in zip(params_data_valid, data_valid):
            # calculate summary statistics
            if flag:
                sum_stats = self.summary[0].calc(datum)  # n_reps x dim stats
                flag = False
            else:
                sum_stats = self.summary[1].calc(datum)  # n_reps x dim stats
                flag = True

            # check validity
            response = self._feedback_summary_stats(sum_stats)
            if response == 'accept' or skip_feedback:
                ret_stats.append(sum_stats)
                # if sum stats is accepted, accept the param as well
                ret_params.append(param)
            elif response == 'discard':
                continue
            else:
                raise ValueError('response not supported')

        return ret_stats, ret_params
