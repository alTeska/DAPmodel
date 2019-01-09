# model based on lfimodels library by Jan-Matthis Lückmann
import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator
from DAPmodel import DAP, DAPExp, DAPBe, DAPFeExp, DAPcython

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

class DAPSimulator(BaseSimulator):
    def __init__(self, I, dt, V0, dim_param=4, prior_log=False, seed=None):
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
        self.I = I
        self.dt = dt
        self.t = np.linspace(0, len(I), len(I))*self.dt

        self.prior_log = prior_log
        self.init = [V0]


    def gen_single(self, params, integration='Backward'):
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

        # if integration == 'Backward':
        #     dap = DAPBe(self.init, params, seed=dap_seed)
        # elif integration == 'Forward':
        #     dap = DAP(self.init, params, seed=dap_seed)
        # elif integration == 'Exponential':
        #     dap = DAPExp(self.init, params, seed=dap_seed)
        # elif integration == 'ForwardExponential':
        #     dap = DAPFeExp(self.init, params, seed=dap_seed)
        # else:
        #     ValueError("No such integration avalibe, try Backward")

        states = dap.simulate(self.dt, self.t, self.I)

        return {'data': states.reshape(-1),
                'time': self.t,
                'dt': self.dt,
                'I': self.I.reshape(-1)}
