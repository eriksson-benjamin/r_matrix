#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 07:47:46 2023

@author: beriksso

Find discharges corresponding to two different groups of CM energies 
corresponding to low and high average CM energy. 

Note
----
This did not work out in the end as the amount of data in the TOF spectrum for
group 1 was too small to do any analysis on, but I will keep this file in the
Github repo for completeness.
"""

import useful_defs as udfs
import numpy as np
import matplotlib.pyplot as plt
udfs.set_nes_plot_style()
from matplotlib.ticker import MaxNLocator
import os


def calculate_weights():
    """
    Calculate the relative contribution of each shot to the TOF spectrum.
    
    Notes
    -----
    Uses TOFu output files in the directory 'path' below to calculate the 
    number of counts in each TOF spectrum. The output is generated using the
    --save-NES argument setting in the TOFu analysis code.
    """
    path = '/common/scratch/beriksso/TOFu/data/r_matrix/nbi_shots/'
    files = os.listdir(path)
    shots = []
    counts = []
    for file in files:
        # Grab shot number
        p = udfs.unpickle(f'{path}/{file}')
        counts.append((p['counts'][499:] - p['bgr_level']).sum())
        shots.append(int(file.split('_')[0]))

    weighted = np.array(counts) / np.max(counts)
    weights = {s: w for s, w in zip(shots, weighted)}
    
    return weights


def plot_lab_frame(j, bin_centres, mask_1, mask_2):
    """Plot the lab frame distributions."""
    plt.figure('Thermal E dist')
    h_th = j['E lab dist'][:, 0]
    plt.plot(bin_centres, h_th[mask_1].T, 'k')
    plt.plot(bin_centres, h_th[mask_2].T, 'r')
    plt.xlabel('$E_{lab}$ (keV)')
    plt.ylabel('Fuel ion density (a.u.)')
    
    plt.figure('NB E dist')
    h_nb = j['E lab dist'][:, 1]
    plt.plot(bin_centres, h_nb[mask_1].T, 'k')
    plt.plot(bin_centres, h_nb[mask_2].T, 'r')
    plt.xlabel('$E_{lab}$ (keV)')
    plt.ylabel('Fuel ion density (a.u.)')


def plot_cm_frame(j, bin_centres, weighted_av, mask_1, mask_2):
    """Plot the CM frame distributions (with and without adjusting for XS)."""
    plt.figure('CM E dist')
    hcm = j['E CM dist'][:, 0]
    plt.plot(bin_centres, hcm[mask_1].T, 'k')
    plt.plot(bin_centres, hcm[mask_2].T, 'r')
    plt.xlabel('$E_{CM}$ (keV)')
    plt.ylabel('Rel. intensity (a.u.)')
    
    plt.figure('CM E (xs) dist')
    hcm_xs = j['E CM dist'][:, 1]
    plt.plot(bin_centres, hcm_xs[mask_1].T, 'k')
    plt.plot(bin_centres, hcm_xs[mask_2].T, 'r')
    plt.plot(bin_centres, weighted_av, 'C0', label='weighted average')
    plt.plot(bin_centres, hcm_xs.mean(axis=0), 'C2', label='average')
    plt.xlabel('Adjusted $E_{CM}$ (keV)')
    plt.ylabel('Rel. intensity (a.u.)')
    plt.legend()


def weighted_distribution(j):
    """
    Calculate the weighted average distribution.
    
    Notes
    -----
    Weights are given by relative contribution to TOF spectrum.
    """
    # Calculate weights
    weights = calculate_weights()
    
    weighted_cm = np.zeros(len(j['bin_centres']))
    for hcm, shot in zip(j['E CM dist'][:, 1, :], j['shots']):
        weighted_cm += hcm * weights[shot]
    
    # Weighted average
    weighted_av = weighted_cm / np.sum(list(weights.values()))

    return weighted_av


def calculate_median(dist, bin_centres):
    """Calculate the 50% +/- 1 sigma limits."""
    # Calculate 1 sigma intervals
    cs = np.cumsum(dist) / dist.sum()
    mean = bin_centres[np.argmin(np.abs(cs - 0.5))]
    std_l = bin_centres[np.argmin(np.abs(cs - 0.15865))]
    std_u = bin_centres[np.argmin(np.abs(cs - (1 - 0.15865)))]
    
    return mean, std_l, std_u
    

def calculate_std(dist, bin_centres):
    """Calculate average and standard deviation of discrete distribution."""
    # Normalize to 1
    normed = dist / np.trapz(dist)
    
    # Calculate average
    mu = np.sum(normed * bin_centres)
    
    # Calculate standard deviation
    sigma = np.sqrt(np.sum((bin_centres - mu)**2 * normed))
    
    return mu, sigma


def calculate_cm_averages(j, mask_1, mask_2, weighted_av):
    # Print average CM energies
    Ecm_xs = j['mean'][:, 0]
    std_lo = j['mean'][:, 1]
    std_hi = j['mean'][:, 2]
    
    # Group 1 (low energy)
    mean_1 = Ecm_xs[mask_1].mean()
    sigma_lo_1 = mean_1 - std_lo[mask_1].mean()
    sigma_hi_1 = std_hi[mask_1].mean() - mean_1
    print('Group 1')
    print('-------')
    print(f'{mean_1:.2f} +({sigma_hi_1:.2f}) -({sigma_lo_1:.2f})')
    print(f'{mean_1 - sigma_lo_1:.2f} < CM E < {mean_1 + sigma_hi_1:.2f}')
    print('')
    
    # Group 2 (high energy)
    mean_2 = Ecm_xs[mask_2].mean()
    sigma_lo_2 = mean_2 - std_lo[mask_2].mean()
    sigma_hi_2 = std_hi[mask_2].mean() - mean_2
    print('Group 2')
    print('-------')
    print(f'{mean_2:.2f} +({sigma_hi_2:.2f}) -({sigma_lo_2:.2f})')
    print(f'{mean_2 - sigma_lo_2:.2f} < CM E < {mean_2 + sigma_hi_2:.2f}')
    print('')
    
    # NBI weighted average
    mu_0, sigma_0 = calculate_std(weighted_av, j['bin_centres'])
    print('')
    print('Weighted distribution')
    print('---------------------')
    print(f'{mu_0:.2f} +/- {sigma_0:.2f}')
    
    
    # NBI non-weighted average
    unweighted_av = (j['E CM dist'][:, 1]).mean(axis=0)
    mu_x, sigma_x = calculate_std(unweighted_av, j['bin_centres'])
    print('')
    print('Un-weighted distribution')
    print('---------------------')
    print(f'{mu_x:.2f} +/- {sigma_x:.2f}')

    means = [(mu_0, sigma_0, sigma_0), (mean_1, sigma_lo_1, sigma_hi_1), 
             (mean_2, sigma_lo_2, sigma_hi_2), (mu_x, sigma_x, sigma_x)]
    
    return means


def plot_averages(means):
    """Plot the CM average energies for the different groups."""
    # Plot the overlap in average CM energy
    plt.figure('Group 1/2 CM energy')
    plt.errorbar(means[1][0], 1, xerr=[[means[1][1]], [means[1][2]]], 
                 marker='.', label='Group 1')
    plt.errorbar(means[2][0], 2, xerr=[[means[2][1]], [means[2][2]]], 
                 marker='.', label='Group 2')
    plt.errorbar(means[0][0], 3, xerr=[[means[0][1]], [means[0][2]]], 
                 marker='.', label='NBI weighted average')
    plt.errorbar(means[3][0], 4, xerr=[[means[0][1]], [means[0][2]]], 
                 marker='.', label='NBI non-weighted average')
    
    plt.xlabel('Adjusted $E_{CM}$ (KeV)')
    plt.ylim(0, 4.5)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()

if __name__ == '__main__':
    # Import CM/lab energy distributions
    cm_dists = 'output_files/cm_distributions/energy_distributions.json'
    j = udfs.numpify(udfs.json_read_dictionary(cm_dists))
    
    # Plot histogram of the average CM energies
    plt.figure('Average CM energy')
    plt.hist(j['mean'][:, 0], bins=np.arange(30, 70, 1))
    plt.xlabel('$E_{CM}$ (keV)')
    plt.ylabel('Frequency')  
    
    bin_centres = j['bin_centres']    
    
    # Calculate the weights from contribution to tof spectrum
    weighted_av = weighted_distribution(j)
    
    # Select distributions from two groups (high and low average CM energy)
    bins = np.array([])
    for h_th, h_nb in j['E lab dist']:
        # Select counts > 0 in NB distribution
        mask_0 = h_nb > 0
        
        # Save final non zero bin (end of the tail of NB lab energy dist.)
        bins = np.append(bins, bin_centres[mask_0].max())

    # Create masks for two groups (tails ending above 122 and below 110 keV)
    mask_1 = bins < 110
    mask_2 = bins > 122
    
    # Plot the two groups in the lab frame
    plot_lab_frame(j, bin_centres, mask_1, mask_2)
    
    # Plot the two groups in the CM frame
    plot_cm_frame(j, bin_centres, weighted_av, mask_1, mask_2)
    
    # Plot the average CM energies
    means = calculate_cm_averages(j, mask_1, mask_2, weighted_av)
    plot_averages(means)

    # Output list of shots from group 1/2
    shot_list = np.loadtxt('../data/nbi/shots_nbi.txt')
    shots = np.array(shot_list[:, 0], dtype='int')
    t0 = shot_list[:, 1]
    t1 = shot_list[:, 2]
    
    group_1 = []
    for s in j['shots'][mask_1]:
        arg = np.argwhere(s == shots)[0][0]
        group_1.append([int(shots[arg]), t0[arg], t1[arg]])
    
    group_2 = []
    for s in j['shots'][mask_2]:
        arg = np.argwhere(s == shots)[0][0]
        group_2.append([int(shots[arg]), t0[arg], t1[arg]])
    
    with open('group_1.txt', 'w') as handle:
        for g1 in group_1:
            handle.write(f'{int(g1[0])} {round(g1[1], 3)} {round(g1[2], 3)}\n')
            
    with open('group_2.txt', 'w') as handle:
        for g2 in group_2:
            handle.write(f'{int(g2[0])} {round(g2[1], 3)} {round(g2[2], 3)}\n')


