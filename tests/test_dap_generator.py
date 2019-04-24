import numpy as np
import matplotlib.pyplot as plt

from delfi.distribution import Uniform
from delfi.inference import SNPE
# from delfi.generator import Default
# from dap import DAPcython
from dap.dap_sim_multi_protocol import DAPSimulatorMultiProtocol
from dap.dap_sumstats_moments import DAPSummaryStatsMoments
from dap.dap_sumstats_step_mom import DAPSummaryStatsStepMoments
from dap.dap_generator import DAPDefault
from dap.utils import obs_params, load_current, load_prior_ranges

np.set_printoptions(suppress=True, precision=2)


dt = 0.01
reg_lambda = 0.01
params, labels = obs_params()
params_list = [params]
data_dir = '/home/alteska/Desktop/LFI_DAP/data/rawData/2015_08_26b.dat'    # best cell

# load the data
Ir, vr, tr, t_onr, t_offr, dtr = load_current(data_dir, protocol='rampIV', ramp_amp=3.1)
Is, vs, ts, t_ons, t_offs, dts = load_current(data_dir, protocol='IV', ramp_amp=1)
I_all = [Ir, Is]
dt_all = [dtr, dts]

# generate data format for SNPE / OBSERVABLE
x_ramp = {'data': vr.reshape(-1),
# x_ramp = {'data': vr,
          'time': tr,
          'dt': dtr,
          'I': Ir}

x_step = {'data': vs.reshape(-1),
# x_step = {'data': vs,
          'time': ts,
          'dt': dts,
          'I': Is}


# Setup Priors
prior_min, prior_max, labels = load_prior_ranges(2)
prior_unif = Uniform(lower=prior_min, upper=prior_max)

# define model
dap1 = DAPSimulatorMultiProtocol(I_all, dt_all, -75)

# run model
stats = dap1.gen_single(params, I_all[0], tr, dtr)
data_list = dap1.gen(params_list)

# calcualte summary statistics
sum_stats_step = DAPSummaryStatsStepMoments(t_ons, t_offs, n_summary=17)
sum_stats_mom = DAPSummaryStatsMoments(t_onr, t_offr, n_summary=17)
sum_stats = [sum_stats_step, sum_stats_mom]

s_step = sum_stats_step.calc([x_step])
s_ramp = sum_stats_mom.calc([x_ramp])
# S = [s_step, s_ramp]
S = s_step
print('summary starts observed:',S)

print('summary stats ramp:', sum_stats_mom.calc(data_list[0]))
print('summary stats step:', sum_stats_step.calc(data_list[1]))



G = DAPDefault(model=dap1, prior=prior_unif, summary=sum_stats)  # Generator

inf_snpe = SNPE(generator=G, n_components=1, n_hiddens=[2], obs=S,
                reg_lambda=reg_lambda, pilot_samples=0)

logs, tds, posteriors = inf_snpe.run(n_train=[10], n_rounds=1,
                                     proposal=prior_unif)

print('goal parameters:', params[:2])
print('posterior parameters:', posteriors[-1].mean)

# plot inputs
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))
ax[0].plot(Ir)
ax[1].plot(Is)

# plot voltage traces
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))
ax[0].plot(data_list[0][0]['data'])
ax[1].plot(data_list[1][0]['data'])

# plt.show()
