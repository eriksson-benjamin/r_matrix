# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:01:06 2022

@author: bener807
"""

import corner
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, 'C:/python/useful_definitions/')
import useful_defs as udfs
udfs.set_nes_plot_style()

# Load MCMC samples from file
file_name = '../output_files/mcmc/nbi/mcmc_output.json'
mcmc = udfs.numpify(udfs.json_read_dictionary(file_name))

test_stat = -mcmc['test_stat']
plt.figure('Test statistic')
plt.plot(np.arange(0, len(test_stat)), test_stat)
plt.xlabel('Iteration')
plt.ylabel('C-stat')

# Select samples from which test statistic has stabilized
n_stable = 20000
norm = 35

# Select mask for C-stat
c_max = 203
mask = test_stat[n_stable:] < 203

# Delete the secondary A1/2+ feed parameter (it is always zero)
samples = np.delete(mcmc['feed'][n_stable:], 1, axis=1) / norm
n_pars = samples.shape[1]

# Create corner plot
labels = ['$A_{1/2^+}^{(0)}$', '$A_{1/2^-}^{(0)}$', '$A_{1/2^-}^{(1)}$',
          '$A_{3/2^-}^{(0)}$', '$A_{3/2^-}^{(1)}$', '$A_{nn}$']
corner.corner(samples[mask], show_titles=True, labels=labels, title_fmt='.4f',
              plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])

# Return quantiles
quantiles = np.zeros([n_pars, 3])
for i in range(n_pars):
    quantiles[i] = corner.quantile(samples[:, i], [0.16, 0.5, 0.84])
