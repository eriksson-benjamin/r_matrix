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
file_name = 'output_files/mcmc/fit_16_mcmc.pickle'
mcmc = udfs.unpickle(file_name)
samples = mcmc['samples']
test_stat = mcmc['test_stat']
n_pars = samples.shape[1]

# Create corner plot
labels = ['$A_{1/2^+}^{(0)}$', '$A_{1/2^+}^{(1)}$', 
          '$A_{1/2^-}^{(0)}$', '$A_{1/2^-}^{(1)}$',
          '$A_{3/2^-}^{(0)}$', '$A_{3/2^-}^{(1)}$', '$A_{nn}$']
corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True,
              quantiles=[0.16, 0.5, 0.84], title_fmt='.4f')

# Return quantiles
quantiles = np.zeros([n_pars, 3])
for i in range(n_pars):
    quantiles[i] = corner.quantile(samples[:, i], [0.16, 0.5, 0.84])


# TODO: Calculate branching ratio and uncertainty MCMC output
#
# u_bt_tt = (quantiles[2][2] - quantiles[0]) / 2
# u_gs_tt = (c_gs_tt[1] - c_gs_tt[0]) / 2
#
# br_gs = I_gs_tt / (I_bt_tt + I_gs_tt)
#
# # Propagate uncertainties
# term_1 = I_bt_tt / (I_gs_tt + I_bt_tt)**2 * u_gs_tt
# term_2 = I_gs_tt / (I_gs_tt + I_bt_tt)**2 * u_bt_tt
# u_br_gs = np.sqrt(term_1**2 + term_2**2)
