    # model based on lfimodels library by Jan-Matthis Lückmann
import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator
from dap import DAPcython

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

class DAPSimulatorMultiProtocol(BaseSimulator):
    def __init__(self, I_all, dt_all, V0, dim_param=4, prior_log=False, seed=None):
        """
        DAP simulator

        Parameters
        ----------
        I : array
            Numpy array with the input I
        dt : float
            Timestep
        V0 : float
            Voltage at first time step
        seed : int or None
            If set, randomness across runs is disabled
        """
        super().__init__(dim_param=dim_param, seed=seed)
        self.I_all = I_all
        self.dt_all = dt_all

        self.t_all = []
        for ii, I in enumerate(self.I_all):
            t = np.linspace(0, len(I), len(I))*dt_all[ii]
            self.t_all.append(t)

        self.prior_log = prior_log
        self.init = [V0]


    def gen(self, params_list, n_reps=1, pbar=None):
        """Forward model for simulator for list of parameters

        Parameters
        ----------
        params_list : list of lists or 1-d np.arrays
            List of parameter vectors, each of which will be simulated
        n_reps : int
            If greater than 1, generate multiple samples given param
        pbar : tqdm.tqdm or None
            If None, will do nothing. Otherwise it will call pbar.update(1)
            after each sample.

        Returns
        -------
        data_list : list of lists containing n_reps dicts with data
            Repetitions are runs with the same parameter set, different
            repetitions. Each dictionary must contain a key data that contains
            the results of the forward run. Additional entries can be present.
        """
        data_list = []
        for ii, I in enumerate(self.I_all):
            t = self.t_all[ii]
            dt = self.dt_all[ii]

            for param in params_list:
                rep_list = []
                for r in range(n_reps):
                    rep_list.append(self.gen_single(param, I, t, dt))
                data_list.append(rep_list)
                if pbar is not None:
                    pbar.update(1)

        return data_list

    def gen_single(self, params, I, t, dt):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector
        integration : string deciding on the type of integration used

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        params = param_invtransform(self.prior_log,np.asarray(params))
        assert params.ndim == 1, 'params.ndim must be 1'

        dap_seed = self.gen_newseed()

        dap = DAPcython(self.init, params, seed=dap_seed)
        states = dap.simulate(dt, t, I)

        return {'data': states.reshape(-1),
                'time': t,
                'dt': dt,
                'I': I.reshape(-1)}
