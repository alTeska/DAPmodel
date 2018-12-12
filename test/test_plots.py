import numpy as np
import matplotlib.pyplot as plt
from DAPmodel.utils_analysis import plot_distr, plot_distr_multiple

means = np.array([1,2])
variancs = np.array([1,1])
labels = ['a', 'b']


plot_distr_multiple(means, variancs, labels)
plt.show()
