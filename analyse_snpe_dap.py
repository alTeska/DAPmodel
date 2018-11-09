# Script for Cluster data analysis of GLM -  plots of weights and biases
import os
import argparse
from DAPmodel.utils_analysis import logs_to_plot
from delfi.utils.io import load_pkl
# from tqdm import tqdm   #TODO

# Setting up the enviroment
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="file name")
args = parser.parse_args()

if args.name is None: args.name = ''

# Make directories
direct_inp = 'pickle/dap_model' + args.name + '/'
direct_out = 'plots/dap_models' + args.name + '/'

if not os.path.exists(direct_out):
    print('creating output directory')
    os.makedirs(direct_out)

# Load From Pickle
print('Load From Pickle')
logs = load_pkl(direct_inp + 'dap_logs' + args.name)

# Create Weights Plots
print('Generating Plots')
g_loss = logs_to_plot(logs, 'loss')
# g_weightW = logs_to_plot(logs, 'weights.mW', melted=True)
g_meansW0 = logs_to_plot(logs, 'means.mW0', melted=True)
# g_meansW1 = logs_to_plot(logs, 'means.mW1', melted=True)
g_precisionsW0 = logs_to_plot(logs, 'precisions.mW0', melted=True)
# g_precisionsW1 = logs_to_plot(logs, 'precisions.mW1', melted=True)
g_h1W = logs_to_plot(logs, 'h1.mW', melted=True)

# Create Biases Plots
# g_weightsb = logs_to_plot(logs, 'weights.mb', melted=True)
g_meansb0 = logs_to_plot(logs, 'means.mb0', melted=True)
# g_meansb1 = logs_to_plot(logs, 'means.mb1', melted=True)
g_precisionsb0 = logs_to_plot(logs, 'precisions.mb0', melted=True)
# g_precisionsb1 = logs_to_plot(logs, 'precisions.mb1', melted=True)
g_h1b = logs_to_plot(logs, 'h1.mb', melted=True)

# Save Plots
print('Saving Plots')
g_loss.savefig(direct_out + 'loss.png', bbox_inches='tight')
# g_weightW.savefig(direct_out + 'weightW.png', bbox_inches='tight')
g_meansW0.savefig(direct_out + 'meansW0.png', bbox_inches='tight')
# g_meansW1.savefig(direct_out + 'meansW1.png', bbox_inches='tight')
g_precisionsW0.savefig(direct_out + 'precisionsW0.png', bbox_inches='tight')
# g_precisionsW1.savefig(direct_out + 'precisionsW1.png', bbox_inches='tight')
g_h1W.savefig(direct_out + 'h1W.png', bbox_inches='tight')

# g_weightsb.savefig(direct_out + 'weightsb.png', bbox_inches='tight')
g_meansb0.savefig(direct_out + 'meansb0.png', bbox_inches='tight')
# g_meansb1.savefig(direct_out + 'meansb1.png', bbox_inches='tight')
g_precisionsb0.savefig(direct_out + 'precisionsb0.png', bbox_inches='tight')
# g_precisionsb1.savefig(direct_out + 'precisionsb1.png', bbox_inches='tight')
g_h1b.savefig(direct_out + 'h1b.png', bbox_inches='tight')
