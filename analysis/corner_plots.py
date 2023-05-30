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
name = 'nbi'
file_name = f'../output_files/mcmc/{name}/mcmc_output_45keV.json'
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
c_max = 190
mask = test_stat[n_stable:] < c_max


# Delete the secondary A1/2+ feed parameter (it is always zero)
samples = np.delete(mcmc['feed'][n_stable:], 1, axis=1)

# Mask and divide by norm factor
samples = samples[mask] / norm

# Create corner plot
labels = ['$A_{1/2^+}^{(1)}$', '$A_{1/2^-}^{(1)}$', '$A_{1/2^-}^{(2)}$',
          '$A_{3/2^-}^{(1)}$', '$A_{3/2^-}^{(2)}$', '$A_{nn}$']
fig = corner.corner(samples, show_titles=True, labels=labels,
                    title_fmt='.2f', plot_datapoints=True)

# Return quantiles
n_pars = samples.shape[1]
quantiles = np.zeros([n_pars, 3])
for i in range(n_pars):
    quantiles[i] = corner.quantile(samples[:, i], [0.16, 0.5, 0.84])

# Load NIF feeding factors
p16 = np.loadtxt('input_files/feed_pars/p0_16keV.txt', usecols=1) / 35
p36 = np.loadtxt('input_files/feed_pars/p0_36keV.txt', usecols=1) / 35
p50 = np.loadtxt('input_files/feed_pars/p0_50keV.txt', usecols=1) / 35

# Statistical uncertainties in NIF feeding factors
up16 = np.array([1.6, 0, 0.1, 7, 0.03, 0.2])
up36 = np.array([1.6, 0.2, 8, 0.04, 4.1, 0.2])
up50 = np.array([1.1, 0.1, 5, 0.03, 3, 0.2])

# Overlay lines
corner.overplot_lines(fig, np.delete(p16, 1), color='C0', linestyle='-.')
corner.overplot_lines(fig, np.delete(p36, 1), color='C1', linestyle='dotted')
corner.overplot_lines(fig, np.delete(p50, 1), color='C2',
                      linestyle=udfs.get_linestyle('loosely dashed'))

# Overlay points
p16 = np.expand_dims(np.delete(p16, 1), 0)
p36 = np.expand_dims(np.delete(p36, 1), 0)
p50 = np.expand_dims(np.delete(p50, 1), 0)

corner.overplot_points(fig, p16, color='C0', markersize=8)
corner.overplot_points(fig, p36, color='C1', markersize=8)
corner.overplot_points(fig, p50, color='C2', markersize=8)

# Set limits
axes = np.array(fig.get_axes()).reshape([6, 6])

# x-limits
x0 = (-30, -22.5, -370, 7.5, 200, None)
x1 = (-11, -16.0, -100, 10.5, 550, None)
for i, col in enumerate(axes.T):
    # Add 50th percentile
    col[i].axvline(quantiles[i][1], color='k', linestyle='--')

    # Add fill between 16-84 percintiles
    col[i].fill_between(x=quantiles[i], y1=3 * [0],
                        y2=3 * [sys.float_info.max], color='k', alpha=0.2)

    # Set x-limits
    for ax in col[i:]:
        ax.set_xlim(x0[i], x1[i])


# y-limits
y0 = (None, -23.0, -360, 7.6, 200, 12)
y1 = (None, -16.0, -100, 10.5, 530, 20.0)
for i, row in enumerate(axes):
    for ax in row[:i]:
        ax.set_ylim(y0[i], y1[i])
