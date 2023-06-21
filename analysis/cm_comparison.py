#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:35:35 2023

@author: beriksso

Comparison of neutron emission energy distributions found using TOFu and NIF
data from:
Johnson, M. Gatu, et al. 
"Experimental Evidence of a Variant Neutron Spectrum from the T (t, 2 n) α 
Reaction at Center-of-Mass Energies in the Range of 16–50 keV." 
Physical review letters 121.4 (2018): 042501.
"""

import numpy as np
import useful_defs as udfs
udfs.set_nes_plot_style()
import sys
sys.path.insert(0, '../')
import rmatspec as rms
import matplotlib.pyplot as plt
from find_cm_groups import calculate_std


def normalize(e_ax, e_specs):
    """Return normalization factors w.r.t. e_specs[0]."""
    d = []    
    for i in range(1, len(e_specs)):
        d.append(udfs.normalize((e_ax, e_specs[i]), (e_ax, e_specs[0])))
        
    return d


def calculate_gatu_dists():
    """Calculate the average Ecm directly from M. Gatu's ion distributions."""
    # Load data files
    energies = [3.6, 11.1, 18.3]
    file_path = 'input_files/gatu_distributions/'
    
    for energy in energies:
        e_ax, intensity = np.loadtxt(f'{file_path}/gatu_{energy}keV.txt', 
                                     unpack=True)
    
        # Calculate mean and standard deviation
        mu, sigma = calculate_std(intensity, e_ax)
    
        print(f'Ti = {energy} keV')
        print('------------')
        print(f'Ecm = {mu:.2f} +/- {sigma:.2f}')
        print('')
    
        plt.figure(f'Ti = {energy} keV')
        plt.plot(e_ax, intensity)
        plt.xlabel('Ion effective energy (keV)')
        plt.ylabel('Intensity')
        plt.axvline(mu, linestyle='--', color='k')
        plt.axvline(mu - sigma, linestyle='--', color='k')
        plt.axvline(mu + sigma, linestyle='--', color='k')
    
if __name__ == '__main__':
    # Calculate the average Ecm directly from M. Gatu's data
    calculate_gatu_dists()

    # Name of data set    
    names = ['nbi']
    
    # Load M. G. Johnson feeding parameters
    p_16keV = np.loadtxt('input_files/feed_pars/p0_16keV.txt', usecols=1)
    p_36keV = np.loadtxt('input_files/feed_pars/p0_36keV.txt', usecols=1)
    p_50keV = np.loadtxt('input_files/feed_pars/p0_50keV.txt', usecols=1)
    
    # Generate energy spectra
    exe = '../fortran/run_fortran'
    e_ax, s_16keV, _, _, _, _ = rms.generate_tt_spec(exe, p_16keV).T
    e_ax, s_36keV, _, _, _, _ = rms.generate_tt_spec(exe, p_36keV).T
    e_ax, s_50keV, _, _, _, _ = rms.generate_tt_spec(exe, p_50keV).T
    
    # Use this to normalize y-axis to 1.
    norm = np.max([s_16keV.max(), s_36keV.max(), s_50keV.max()]) / 0.95
    
    # Plot
    plt.figure('Comparison')
    colors = udfs.get_colors(len(names))
    for name, color in zip(names, colors):
        # Load TOFu specs
        spec_path = f'output_files/specs/{name}/high_res_specs.json'
        specs = udfs.json_read_dictionary(spec_path)
        e_specs = np.array(specs['E specs'])[:, 0, :]
        e_std = np.std(e_specs, axis=0)
        e_mean = e_specs.mean(axis=0)
        
        # Normalize w.r.t. s_16keV (NIF spectrum)
        mask = ((e_ax > 4000) & (e_ax < 7000))
        d_norm = normalize(e_ax[mask], (s_16keV[mask], e_mean[mask]))
    
        # Plot TOFu energy spectra
        plt.plot(specs['x_E'], d_norm * e_mean / norm, color, label='$E_{CM} = 44.9$ keV')
        plt.fill_between(specs['x_E'], d_norm * (e_mean - 2*e_std) / norm,
                         d_norm * (e_mean + 2*e_std) / norm, color=color, alpha=0.4)
    
    
    # Normalize w.r.t. s_16keV (NIF spectrum)
    d_norms = normalize(e_ax[mask], (s_16keV[mask], s_36keV[mask], s_50keV[mask]))
    
    # Plot NIF energy spectra
    plt.plot(e_ax, s_16keV / norm, 'C0-.', linewidth=1.5, label='$E_{CM} = 18.5 $ keV')
    plt.plot(e_ax, d_norms[0] * s_36keV / norm, 'C1', linewidth=2.0, linestyle='dotted', 
             label='$E_{CM} = 45.1$ keV')
    plt.plot(e_ax, d_norms[1] * s_50keV / norm, 'C2--', linewidth=1.5, 
             label='$E_{CM} = 58.6$ keV')
    plt.xlabel('$E_n$ (keV)')
    plt.ylabel('Rel. intensity (a.u.)')
    plt.ylim(0, 1)
    plt.xlim(0, 1E4)
    plt.gca().ticklabel_format(style='plain', axis='x')
    
    plt.legend()
        


