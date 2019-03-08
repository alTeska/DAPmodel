# DAP neuron model
DAP (depolarizing afterpotentials) neuron model created for a project at [Prof. Herz lab for Computational Neuroscience](http://www.neuro.bio.lmu.de/members/comp_neuro_herz/herz_a/index.html) at LMU.

#### Project Status: [Active]

## Project Intro

The purpose of this project is creation of a fast and reliable cell model for inference of real data recorded from entorhinal cortex.

### Technologies:
* python
* cython
* [delfi](https://github.com/mackelab/delfi)
* theano
* pandas, jupyter


## Project Description
The model is based on differential equations from classic Hodgkin&Huxley model. The cell has 4 uniformly distributed Ion Channels: Kdr, Na_p, Na_t, Hcn; that account for lack of typical hyperpolarization after an action potential. Three types of Euler integration have been provided: forward, backward and exponential. Due to stability issue with different parameters choices the last integration is used for all of the experiments.

The repo has additional object to ease the integration into inference methods: <i>Simulator, Summary Statistics</i>.

Experimental data is not attached to this repository, however it contains multiple cell recordings with variety of 3 currents: <i>ramp, step</i> and <i>Zap20</i> (changing its frequency sinusoid).

## Data Analysis and Inference
The model stored in this repository is connected to the application of Data Inference onto it, in order to find best fit of parameters that can represent the real cell recorded data. Used scripts can be found in [DAP_analysis](https://github.com/alTeska/DAP_analysis) repository.

## Example of Use

```
from dap import DAPcython
from dap.utils import obs_params_gbar, obs_params, syn_current


dt = 1e-2
params, labels = obs_params_gbar()
I, t, t_on, t_off = syn_current(duration=120, dt=dt)

# define models
dap = DAPcython(-75, params, solver=1)

# run models
U = dap.simulate(dt, t, I)
```
## Currents Visualization
<figure>
<p align="center">
    <img src="/img/ramp.png" width="50%">
    <img src="/img/step_current.png" width="50%">
</p>
</figure>

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Jan-Matthis Lueckmann](https://github.com/jan-matthis) - a lot of files are inspired and built on delfi library models
* [Caroline Fischer](https://github.com/cafischer) - initiating the cell model and writitng the code of which a simplified version is used in dap.cell_fitting
