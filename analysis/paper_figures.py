#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:43:52 2023

@author: beriksso
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import useful_defs as udfs
import param_analysis as pa
import sys
import find_cm_groups as fcg
import calculate_CM_energy as cce


def plot_tof(name, n_specs, tof_path, specs):
    """Plot TOF spectrum from file."""
    # Read TOF data
    tof_x, tof_y, tof_bgr = np.loadtxt(tof_path, delimiter=',', unpack=True)

    # Read DT and scatter
    bt_td = udfs.json_read_dictionary(f'../input_files/specs/{name}/bt_td_spec.json')
    scatter = udfs.json_read_dictionary(f'../input_files/specs/{name}/scatter_spec.json')
    tof_dt = np.array(bt_td['tof']['y'])
    tof_sc = np.array(scatter['tof']['y'])

    # Read specs
    spec_path = f'output_files/specs/{name}/{specs}'
    specs = udfs.numpify(udfs.json_read_dictionary(spec_path))
    erg_specs = specs['E specs']
    tof_specs = specs['tof specs']
    
    # Create mask for selecting spectra
    mask = np.random.choice(np.arange(0, len(erg_specs)), n_specs, 
                            replace=False)
    
    # Select spectra
    erg_tot, erg_01, erg_02, erg_03, erg_nn, erg_in = np.transpose(erg_specs[mask], 
                                                           (1, 0, 2))
    tof_tot, tof_01, tof_02, tof_03, tof_nn, tof_in = np.transpose(tof_specs[mask], 
                                                           (1, 0, 2))
    """
    Plot TOF spectra
    """
    # Create figure
    plt.figure('TOF fig 1')
    ax1 = plt.gca()

    # Data
    ax1.errorbar(tof_x, tof_y, yerr=np.sqrt(tof_y), linestyle='None',
                 color='k', marker='.', markersize=1)

    alpha = 1
    

    # DT
    ax1.plot(tof_x, tof_dt, color='b', label='DT',
             linestyle=udfs.get_linestyle('long dash with offset'))
    
    # Scatter
    ax1.plot(tof_x, tof_sc, 'k-.', label='scatter')

    # TT total
    ax1.plot(tof_x, tof_tot.T, label='TT', color='C1', 
             linestyle='-', alpha=alpha)

    # Background
    ax1.plot(tof_x, tof_bgr, 'C2--', label='background')

    legend_el = [Line2D([0], [0], color='b', label='DT', 
                        linestyle=udfs.get_linestyle('long dash with offset')),
                 Line2D([0], [0], color='C1', label='TT', 
                        linestyle='-'),
                 Line2D([0], [0], color='k', label='scatter', 
                        linestyle='-.'),
                 Line2D([0], [0], color='C2', label='background', 
                        linestyle='--')]
    ax1.legend(handles=legend_el)
    ax1.set_xlabel('$t_{TOF}$ (ns)')
    ax1.set_ylabel('counts')


    ax1.set_xlim(10, 120)
    ax1.set_ylim(10, 20000)
    ax1.set_yscale('log')


def plot_E(name, specs):
    """Plot two energy spectra to exemplify covariance in feeding factors."""
    # Read specs
    spec_path = f'output_files/specs/{name}/{specs}'
    specs = udfs.numpify(udfs.json_read_dictionary(spec_path))
    erg_specs = specs['E specs']
    
    # Create mask for selecting spectra
    np.random.seed(17)
    mask = np.random.choice(np.arange(0, len(erg_specs)), 2, replace=False)
    
    # Select spectra
    erg_tot, erg_01, erg_02, erg_03, erg_nn, erg_in = np.transpose(erg_specs[mask], 
                                                           (1, 0, 2))

    """
    Plot energy spectra
    """
    # Create figures
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.canvas.manager.set_window_title('Energy fig 1') 

    fig.set_size_inches(5, 9)
    plt.subplots_adjust(hspace=0.1)
    
    # Figure 1
    # --------
    # Total
    erg_x = specs['x_E']
    alpha = 1
    linestyles = ['-', '--']
    norm = np.max(erg_tot) / 0.75
    for i in range(2):
        ax1.plot(erg_x, erg_tot[i] / norm, color='C1', linestyle=linestyles[i], 
                 label='total', alpha=alpha)

        # 1/2-
        ax1.plot(erg_x, erg_02[i] / norm, label=r'$1/2- \ n \alpha$', 
                 color='k', alpha=alpha, linestyle=linestyles[i])
    
        # 3/2-
        ax1.plot(erg_x, erg_03[i] / norm, label=r'$3/2- \ n \alpha$', 
                 color='C0', alpha=alpha, linestyle=linestyles[i])
    
        # nn
        ax1.plot(erg_x, erg_nn[i] / norm, label=r'nn', color='C2', alpha=alpha, 
                 linestyle=linestyles[i])
    
        # Figure 2
        # --------
        # Total
        
        # TT total
        ax2.plot(erg_x, erg_tot[i] / norm, label='TT total', color='C1',
                 alpha=alpha, linestyle=linestyles[i])
    
        # 1/2̈́+
        ax2.plot(erg_x, erg_01[i] / norm, label=r'$1/2+ \ n \alpha$', 
                 color='C3', alpha=alpha, linestyle=linestyles[i])
    
        # interference
        ax2.plot(erg_x, erg_in[i] / norm, label='interference', color='C4', 
                 alpha=alpha, linestyle=linestyles[i])
    
    # Legends
    legend_el = [Line2D([0], [0], color='C1', label='TT total'),
                 Line2D([0], [0], color='k', label=r'$1/2^- \ n \alpha$'),
                 Line2D([0], [0], color='C0', label=r'$3/2^- \ n \alpha$'),
                 Line2D([0], [0], color='C2', label='nn')]
    ax1.legend(handles=legend_el)
    
    legend_el = [Line2D([0], [0], color='C1', label='TT total'),
                 Line2D([0], [0], color='C3', label=r'$1/2^+ \ n \alpha$'),
                 Line2D([0], [0], color='C4', label='interference')]
    ax2.legend(handles=legend_el)
    
    # Limits
    ax1.set_xlim(-500, 10000)
    ax1.set_ylim(0, 1)

    ax2.set_xlim(-500, 10000)
    ax2.set_ylim(-0.5, 1)

    # Labels
    ax1.set_ylabel('Rel. intensity (a.u.)')
    ax2.set_ylabel('Rel. intensity (a.u.)')
    ax2.set_xlabel('$E_n$ (keV)')
    ax2.ticklabel_format(style='plain', axis='x')
    ax1.text(0.035, 0.9, '(a)', transform=ax1.transAxes)
    ax2.text(0.035, 0.9, '(b)', transform=ax2.transAxes)
    
    
def plot_stats(C_stat, feed_factors, n=1000):
    """
    Plot the feed factors from the MCMC output.

    Parameters
    ----------
    feed_factors : ndarray
        A 2D array containing the feed factors from the MCMC output.
    n : int, optional
        Number of elements (plus/minus) to include in the running average.

    Returns
    -------
    None

    Notes
    -----
    The function plots the feed factors from the MCMC output by calculating the
    running average and standard deviation of each feed factor using a window 
    of size 2n+1. The output is a plot with two subplots, one for each feed 
    factor.
    """
    
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
    fig.canvas.manager.set_window_title('MCMC stats') 

    def plot_A(ax1, ax2, a, b):
        # Calculate running average
        rm_a, rs_a = pa.running_average(feed_factors[:, a] / 35, n)
        rm_b, rs_b = pa.running_average(feed_factors[:, b] / 35, n)
        x_rma = np.arange(0, len(rm_a))
        x_rmb = np.arange(0, len(rm_b))
        
        # Plot
        ax1.plot(x_axis, feed_factors[:, a] / 35, color='C0', 
                 zorder=1, alpha=alpha)
        ax1.plot(x_rma, rm_a, label=labels[a], color='C1', zorder=3)
        ax1.fill_between(x_rma, y1=rm_a + rs_a, y2=rm_a - rs_a,
                         color='xkcd:peach', zorder=2)

        ax2.plot(x_axis, feed_factors[:, b] / 35, color='C0', 
                 alpha=alpha, zorder=1)
        ax2.plot(x_rmb, rm_b, label=labels[b], color='C1', zorder=3)
        ax2.fill_between(x_rmb, y1=rm_b + rs_b, y2=rm_b - rs_b,
                         color='xkcd:peach', zorder=2)

        ax1.set_title(labels[a], loc='left')
        ax2.set_title(labels[b], loc='left')
        fig.set_size_inches(7, 10)
    
    alpha = 0.5
    x_axis = np.arange(0, len(feed_factors))

    labels = ['$A_{1/2^+}^{(1)}$', '$A_{1/2⁺}^{(2)}$', 
              '$A_{1/2^-}^{(1)}$', '$A_{1/2^-}^{(2)}$',
              '$A_{3/2^-}^{(1)}$', '$A_{3/2^-}^{(2)}$', 
              '$A_{nn}$']
    # 3/2-, l=1,2
    plot_A(axes[0, 0], axes[0, 1], 4, 5)
    # 1/2-, l=1,2
    plot_A(axes[1, 0], axes[1, 1], 2, 3)
    # 1/2+, l=1 & nn
    plot_A(axes[2, 0], axes[2, 1], 0, 6)
    
    # x-, y-labels
    fs = 14
    axes[0, 0].set_ylabel('Parameter value', fontsize=fs)
    axes[1, 0].set_ylabel('Parameter value', fontsize=fs)
    axes[2, 0].set_ylabel('Parameter value', fontsize=fs)
    axes[2, 0].set_xlabel('Iteration', fontsize=fs)
    axes[2, 1].set_xlabel('Iteration', fontsize=fs)

    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    y_lims = [(7, 11), (120, 520), (-23, -19), 
              (-350, -100), (-27, -9), (8, 20)]
    for i, ax in enumerate(axes.flatten()):
        ax.text(0.9, 0.9, letters[i], transform=ax.transAxes)
        ax.set_ylim(y_lims[i])
    
    
def plot_Elab(bin_centres, h_th, h_nb):
    """Plot the lab T energy distributions."""
    # Add (0, 0) to thermal distribution
    th_bins = np.insert(bin_centres, 0, 0)
    h_th = np.insert(h_th, 0, 0)
        
    # Plot
    plt.figure('Fuel ion distributions')
    plt.plot(th_bins, h_th / (1.3 * h_th.max()), 'C0--', label='Thermal T')
    plt.plot(bin_centres, h_nb / (1.3 * h_nb.max()), 'C1-', label='NB T')
    plt.xlabel('Fuel ion $E_{lab}$ (keV)')
    plt.ylabel('Fuel density (a.u.)')
    plt.legend()
    
    ax1 = plt.gca()
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-5, 125)
    

def plot_ECM(bin_centres, Ecm_h, Ecm_xs):
    """Plot the CM energy distributions together with T+T cross section."""
    plt.figure('CM energy distributions')
    
    # Plot CM E dist. and cross section adjusted distribution
    plt.plot(bin_centres, Ecm_h / (1.3 * Ecm_h.max()), 'k--', label='T+T')
    plt.plot(bin_centres, Ecm_xs / (1.3 * Ecm_xs.max()), 'k-', 
             label='T+T (adjusted by $\sigma$v)')
    
    plt.xlabel('Fuel ion $E_{CM}$ (keV)')
    plt.ylabel('Rel. intensity (a.u.)')
    ax1 = plt.gca()
    ax1.set_ylim(0, 1)

    # Calculate 1 sigma intervals
    mean, std_l, std_u = fcg.calculate_median(Ecm_xs, bin_centres)

    mu, sigma = fcg.calculate_std(Ecm_xs, bin_centres)

    # Get fuel ion energy distributions
    th, nb = cce.fuel_ion_dists(98694, 47.55, 51.221)

    # Calculate CM energy of the samples 
    Ecm, V = cce.calculate_Ecm(th, nb)
    
    # Sort
    argsort = np.argsort(Ecm)
    Ecm_sorted = Ecm[argsort]
    V_sorted = V[argsort]
    
    # Calculate sigma*v in m^3/s
    sigma = cce.read_xsec(Ecm_sorted, COFM=True)
    sigma_v = sigma * V_sorted

    ax2 = ax1.twinx()
    
    ax2.plot(Ecm_sorted, sigma_v, 'r-.', label='$\sigma$v')
    ax2.set_ylabel('$\sigma$v ($m^3$/s)', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(loc='upper right')
    ax2.set_xlim(-5, 125)
    ax2.set_ylim(0, 2E-23)
    ax1.legend(loc='upper left')


def plot_all_ECM(j, weighted_av):
    """Plot the CM frame distributions (with and without adjusting for XS."""
    plt.figure('Paper - CM E dist')
    hcm = j['E CM dist'][:, 0]
    bin_centres = j['bin_centres']
    plt.plot(bin_centres, hcm.T, 'k', alpha=0.3)
    plt.xlabel('$E_{CM}$ (keV)')
    plt.ylabel('Rel. intensity (a.u.)')
    
    plt.figure('Paper - CM E (xs) dist')
    hcm_xs = j['E CM dist'][:, 1]
    mu, sigma = fcg.calculate_std(weighted_av, bin_centres)
    
    norm = 0.6 / weighted_av.max()
    plt.plot(bin_centres, hcm_xs.T * norm, 'k', alpha=0.3)
    plt.plot(bin_centres, weighted_av * norm , 'C1', label='weighted average', 
             linewidth=1.5)
    plt.axvline(mu, linestyle='--', color='C1', linewidth=1.5)
    
    plt.xlabel('Adjusted $E_{CM}$ (keV)')
    plt.ylabel('Rel. intensity (a.u.)')
    plt.ylim(0, 1)
    plt.xlim(0, 125)
    plt.legend()
    

def plot_E_tof(name, n_specs, tof_path):
    """Plot energy and TOF spectra from file. Same as plot_tof() but faster."""
    # Read TOF data
    tof_x, tof_y, tof_bgr = np.loadtxt(tof_path, delimiter=',', unpack=True)

    # Read DT and scatter
    bt_td = udfs.json_read_dictionary(f'../input_files/specs/{name}/bt_td_spec.json')
    scatter = udfs.json_read_dictionary(f'../input_files/specs/{name}/scatter_spec.json')
    tof_dt = np.array(bt_td['tof']['y'])
    tof_sc = np.array(scatter['tof']['y'])

    # Read specs
    spec_path = f'output_files/specs/{name}/specs.json'
    specs = udfs.numpify(udfs.json_read_dictionary(spec_path))
    erg_specs = specs['E specs']
    tof_specs = specs['tof specs']
    
    # Create mask for selecting spectra
    mask = np.random.choice(np.arange(0, len(erg_specs)), n_specs, 
                            replace=False)
    
    # Select spectra
    erg_tot, erg_01, erg_02, erg_03, erg_nn, erg_in = np.transpose(erg_specs[mask], 
                                                           (1, 0, 2))
    tof_tot, tof_01, tof_02, tof_03, tof_nn, tof_in = np.transpose(tof_specs[mask], 
                                                           (1, 0, 2))
    """
    Plot TOF spectra
    """
    # Create figure
    plt.figure('TOF fig 2')
    ax1 = plt.gca()

    # Data
    ax1.errorbar(tof_x, tof_y - tof_bgr, yerr=np.sqrt(tof_y), linestyle='None',
                 color='k', marker='.', markersize=1)

    alpha = 0.4
    

    # Total
    ax1.plot(tof_x, (tof_dt + tof_tot + tof_sc).T, 'r-', label='total',
             alpha=alpha)

    # DT
    ax1.plot(tof_x, tof_dt, 'b--', label='DT')
    
    # Scatter
    ax1.plot(tof_x, tof_sc, 'k-.', label='scatter')

    # TT total
    ax1.plot(tof_x, tof_tot.T, label='TT', color='C1', 
             linestyle='-', alpha=alpha)


    legend_el = [Line2D([0], [0], color='r', label='total'),
                 Line2D([0], [0], color='b', label='DT', linestyle='--'),
                 Line2D([0], [0], color='C1', label='TT'),
                 Line2D([0], [0], color='k', label='scatter', linestyle='-.')]
    ax1.legend(handles=legend_el)
    ax1.set_xlabel('$t_{TOF}$ (ns)')
    ax1.set_ylabel('counts')


    ax1.set_xlim(30, 100)
    ax1.set_ylim(0, 5000)

    """
    Plot energy spectra
    """
    # Create figures
    plt.figure('Energy fig 2')
    ax1 = plt.gca()

    erg_x = specs['x_E']
    
    # Total
    erg_max = np.max(erg_tot, axis=1)
    ax1.plot(erg_x/1E3, 0.85*(erg_tot.T / erg_max), 'C1-', label='TT total', 
             alpha=alpha)
    ax1.set_xlabel('Neutron energy (MeV)')
    ax1.set_ylabel('intensity (a.u.)')

    legend_el = [Line2D([0], [0], color='C1', label='TT total')]
    ax1.legend(handles=legend_el)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 1)

if __name__ == '__main__':
    # Plot TOF/E specs from files
    name = 'nbi'
    tof_path = f'../data/{name}/{name}.txt'
    specs = 'specs.json'
    plot_tof(name, 1, tof_path, specs)
    plot_E(name, specs)
    plot_E_tof(name, 100, tof_path)
    
    # Read MCMC file
    path = f'../output_files/mcmc/{name}/mcmc_output.json'
    mcmc = udfs.numpify(udfs.json_read_dictionary(path))
    C_stat = -mcmc['test_stat']
    
    # Apply mask
    mask = C_stat < 197
    params = mcmc['feed'][mask]
    C_stat = C_stat[mask]

    # Plot stats
    plot_stats(C_stat, params, n=100)

    # Import CM/lab energy distributions
    cm_dists = 'output_files/cm_distributions/energy_distributions.json'
    j = udfs.numpify(udfs.json_read_dictionary(cm_dists))
    
    # Plot lab frame fuel ion distributions
    n = 25  # This is the distribution with the highest weight to the TOF spec
    print(j['shots'][n])
    h_th = j['E lab dist'][:, 0, :][n]
    h_nb = j['E lab dist'][:, 1, :][n]
    bin_centres = j['bin_centres']    
    plot_Elab(bin_centres, h_th, h_nb)

    # Plot CM fuel ion distributions adjusted by cross section and velocity
    Ecm_h = j['E CM dist'][:, 0, :][n]
        
    # Calculate the weights from contribution to tof spectrum
    weighted_av = fcg.weighted_distribution(j)
    plot_ECM(bin_centres, Ecm_h, weighted_av)
    
    # Plot all sigma*v adjusted distributions
    plot_all_ECM(j, weighted_av)
