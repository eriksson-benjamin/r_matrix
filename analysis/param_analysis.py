#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:11:20 2023

@author: beriksso
"""

import useful_defs as udfs
udfs.set_nes_plot_style()
import numpy as np
import matplotlib.pyplot as plt
import sys
import rmatspec as rms
from nes.tofor.commands import load_response_function
from matplotlib.lines import Line2D


def running_average(a, n):
    """
    Compute the running average and standard deviation of an input array.

    Parameters
    ----------
    a : array_like,
        Input array to compute the running average and standard deviation.
    n : int,
        Number of elements (plus/minus) to include in the running average.

    Returns
    -------
    average : ndarray,
        Array of the same shape as the input array 'a' containing the running
        averages.
    std : ndarray,
        Array of the same shape as the input array 'a' containing the running
        standard deviations.

    Notes
    -----
    The function computes the running average and standard deviation of the
    input array by dividing it into three parts:
    - The beginning and end parts of length 'n', where the running average and
      standard deviation are computed independently.
    - The middle part of the array, where the running average and standard
      deviation are computed over a window of length '2n+1'.
    """
    # Beginning/end part of list
    start_of_list = np.empty([n, 2 * n])
    start_of_list[:] = np.nan

    end_of_list = np.empty([n, 2 * n])
    end_of_list[:] = np.nan

    for i in range(n):
        # Choose beginning/end of list
        start_of_list[i][0:i + n + 1] = a[0:i + n + 1]
        end_of_list[i][-2 * n + i:] = a[-2 * n + i:]

    # Calculate average
    start_average = np.nanmean(start_of_list, axis=1)
    end_average = np.nanmean(end_of_list, axis=1)

    # Calculate standard deviation
    start_std = np.nanstd(start_of_list, axis=1)
    end_std = np.nanstd(end_of_list, axis=1)

    # Middle part of list
    mid_of_list = np.zeros([len(a) - 2 * n, 2 * n + 1])
    for i in range(len(a) - 2 * n):
        # Choose middle part of list
        mid_of_list[i] = a[i:2 * n + i + 1]

    # Calculate average
    mid_average = np.mean(mid_of_list, axis=1)

    # Calculate standard devitation
    mid_std = np.std(mid_of_list, axis=1)

    # Concatenate arrays
    average = np.concatenate((start_average, mid_average, end_average))
    std = np.concatenate((start_std, mid_std, end_std))

    return average, std


def plot_test_stat(C_stat, fig_name, n=1000):
    """
    Plot the C_stat from the MCMC output.

    Parameters
    ----------
    C_stat : array_like,
        The C_stat from the MCMC output.
    fig_name : str,
        The name of the plot figure.
    n : int, optional,
        Number of elements (plus/minus) to include in the running average. 
        Default is 1000.

    Returns
    -------
    None

    Notes
    -----
    The function plots the C_stat from the MCMC output and a running average 
    with a window of length '2n+1'. Additionally, it computes and fills the 
    area between the upper and lower bounds of the standard deviation.
    """
    plt.figure(fig_name)
    plt.plot(np.arange(0, len(C_stat)), C_stat, zorder=1)
    rm, rs = running_average(C_stat, n)
    x_axis = np.arange(0, len(rm))
    plt.plot(x_axis, rm, 'r-', label='running average', zorder=3)
    plt.fill_between(x_axis, y1=rm + rs, y2=rm - rs, color='xkcd:reddish',
                     zorder=2)

    plt.xlabel('Iteration')
    plt.ylabel('C-stat')
    plt.legend()


def plot_feed_factors(feed_factors, n=1000):
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
    def plot_A(fig_name, a, b):
        # Calculate running average
        rm_a, rs_a = running_average(feed_factors[:, a], n)
        rm_b, rs_b = running_average(feed_factors[:, b], n)
        x_rma = np.arange(0, len(rm_a))
        x_rmb = np.arange(0, len(rm_b))

        # Plot
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       num=fig_name)

        ax1.plot(x_axis, feed_factors[:, a], color='C0', zorder=1)
        ax1.plot(x_rma, rm_a, label=labels[a], color='C1', zorder=3)
        ax1.fill_between(x_rma, y1=rm_a + rs_a, y2=rm_a - rs_a,
                         color='xkcd:peach', zorder=2)

        ax2.plot(x_axis, feed_factors[:, b], color='C0', alpha=1, zorder=1)
        ax2.plot(x_rmb, rm_b, label=labels[b], color='C1', zorder=3)
        ax2.fill_between(x_rmb, y1=rm_b + rs_b, y2=rm_b - rs_b,
                         color='xkcd:peach', zorder=2)

        ax1.set_ylabel('Parameter value')
        ax2.set_ylabel('Parameter value')
        ax2.set_xlabel('Iteration')

        ax1.set_title(labels[a], loc='left')
        ax2.set_title(labels[b], loc='left')

        ax1.ticklabel_format(style='plain')
        ax2.ticklabel_format(style='plain')

        fig.set_size_inches(5, 8)

    x_axis = np.arange(0, len(feed_factors))
    labels = [r'$1/2^+ \ A_1$', r'$1/2^+ \ A_2$',
              r'$1/2^- \ A_1$', r'$1/2^- \ A_2$',
              r'$3/2^- \ A_1$', r'$3/2^- \ A_2$', 'nn']

    plot_A('Feed A 1/2+', 0, 1)
    plot_A('Feed A 1/2-', 2, 3)
    plot_A('Feed A 3/2-', 4, 5)

    plt.figure('Feed A nn')
    rm, rs = running_average(feed_factors[:, 6], n)
    x_rm = np.arange(0, len(rm))
    plt.plot(x_axis, feed_factors[:, 6], color='C0', zorder=1)
    plt.plot(x_rm, rm, color='C1', label=labels[6], zorder=3)
    plt.fill_between(x_rm, rm + rs, rm - rs, zorder=2, color='xkcd:peach')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')


def plot_modifiers(modifiers, n=1000):
    """
    Plot the modifiers from the MCMC output.

    Parameters
    ----------
    modifiers : ndarray
        A 2D array containing the modifiers from the MCMC output.
    n : int, optional
        Number of elements (plus/minus) to include in the running average.

    Returns
    -------
    None

    Notes
    -----
    The function plots the modifiers from the MCMC output by calculating the
    running average and standard deviation of each modifier using a window of 
    size 2n+1. The output is a plot with two subplots, one for each feed 
    factor.
    """
    def plot_A(fig_name, a, b):
        # Calculate running average
        rma, rsa = running_average(modifiers[:, a], n)
        rmb, rsb = running_average(modifiers[:, b], n)
        x_rma = np.arange(0, len(rma))
        x_rmb = np.arange(0, len(rmb))

        # Plot
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       num=fig_name)

        ax1.plot(x_axis, modifiers[:, a], color='C0', zorder=1)
        ax1.plot(x_rma, rma, label=labels[a], color='C1', zorder=3)
        ax1.fill_between(x_rma, y1=rma + rsa, y2=rma - rsa, color='xkcd:peach',
                         zorder=2)

        ax2.plot(x_axis, modifiers[:, b], color='C0', alpha=1, zorder=1)
        ax2.plot(x_rmb, rmb, label=labels[b], color='C1', zorder=3)
        ax2.fill_between(x_rmb, y1=rmb + rsb, y2=rmb - rsb, color='xkcd:peach',
                         zorder=2)

        ax1.set_ylabel('Parameter value')
        ax2.set_ylabel('Parameter value')
        ax2.set_xlabel('Iteration')

        ax1.set_title(labels[a], loc='left')
        ax2.set_title(labels[b], loc='left')

        ax1.ticklabel_format(style='plain')
        ax2.ticklabel_format(style='plain')

        fig.set_size_inches(5, 8)

    x_axis = np.arange(0, len(modifiers))
    labels = [r'$1/2^+ \ A_1$', r'$1/2^+ \ A_2$',
              r'$1/2^- \ A_1$', r'$1/2^- \ A_2$',
              r'$3/2^- \ A_1$', r'$3/2^- \ A_2$', 'nn']

    plot_A('Mods A 1/2+', 0, 1)
    plot_A('Mods A 1/2-', 2, 3)
    plot_A('Mods A 3/2-', 4, 5)

    plt.figure('Mods A nn')
    rm, rs = running_average(modifiers[:, 6], n)
    x_rm = np.arange(0, len(rm))
    plt.plot(x_axis, modifiers[:, 6], color='C0', zorder=1)
    plt.plot(x_rm, rm, color='C1', label=labels[6], zorder=3)
    plt.fill_between(x_rm, rm + rs, rm - rs, zorder=2, color='xkcd:peach')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')


def plot_tof(exe, feeds, tof_path, drf_path, rigid_shift, save_specs=False):
    """
    Plot the TOF spectra for a given set of feed factors.

    Parameters
    ----------
    exe : str,
        Path to executable used to generate TT spectra.
    feeds : ndarray,
        2D array of feed factors where each row corresponds to one set of 
        parameters.
    tof_path : str,
        Path to time-of-flight data.
    drf_path : str,
        Path to detector response function.
    rigid_shift : float,
        Shift in ns to apply to modelled TOF spectrum.

    Returns
    -------
    None

    Notes
    -----
    The function generates the TT time-of-flight spectrum for the given sets of
    feed parameters and plots them on top of the experimental TOF data.
    """
    # Read TOF data
    tof_x, tof_y, tof_bgr = np.loadtxt(tof_path, delimiter=',', unpack=True)

    # Read DT and scatter
    bt_td = udfs.json_read_dictionary('input_files/specs/bt_td_spec.json')
    scatter = udfs.json_read_dictionary('input_files/specs/scatter_spec.json')

    # Read response function
    drf = load_response_function(drf_path)

    # Translate to TOF
    tof_dt = bt_td['tof']['y']
    tof_sc = scatter['tof']['y']

    plt.figure('Fig 1')
    ax1 = plt.gca()

    plt.figure('Fig 2')
    ax2 = plt.gca()

    # Data
    ax1.errorbar(tof_x, tof_y - tof_bgr, yerr=np.sqrt(tof_y), linestyle='None',
                 color='k', marker='.', markersize=1)

    ax2.errorbar(tof_x, tof_y - tof_bgr, yerr=np.sqrt(tof_y), linestyle='None',
                 color='k', marker='.', markersize=1)
    alpha = 0.4
    info = ('"E/tof specs" contains neutron spectra produced using the '
            'feeding parameters in "feed params". The order is total, 1/2+, '
            '1/2-, 3/2-, nn, interference. x_E is in MeV. x_tof is in ns.')
    specs = {'feed params': [], 'x_E': [], 'E specs': [],
             'tof specs': [], 'x_tof': [], 'info': info}
    for i, feed in enumerate(feeds):
        print(f'{i+1}/{len(feeds)}')
        f = [str(p) for p in feed]
        # Create TT energy spectrum
        tt_tot, tt_01, tt_02, tt_03, tt_nn = rms.generate_components(exe, f)

        # Translate to TOF
        tof_tot = rms.calculate_tt_tof(
            drf, rigid_shift, tt_tot[:, 0], tt_tot[:, 1])
        tof_01 = rms.calculate_tt_tof(
            drf, rigid_shift, tt_01[:, 0], tt_01[:, 1])
        tof_02 = rms.calculate_tt_tof(
            drf, rigid_shift, tt_02[:, 0], tt_02[:, 1])
        tof_03 = rms.calculate_tt_tof(
            drf, rigid_shift, tt_03[:, 0], tt_03[:, 1])
        tof_nn = rms.calculate_tt_tof(
            drf, rigid_shift, tt_nn[:, 0], tt_nn[:, 1])
        tof_in = rms.calculate_tt_tof(
            drf, rigid_shift, tt_tot[:, 0], tt_tot[:, -1])

        # Figure 1
        # --------
        # Total
        ax1.plot(tof_x, tof_dt + tof_tot + tof_sc, 'r-', label='total',
                 alpha=alpha)

        # TT total
        ax1.plot(tof_x, tof_tot , label='TT total', color='C1',
                 alpha=alpha)

        # 1/2-
        ax1.plot(tof_x, tof_02, label=r'$1/2- \ n \alpha$', color='k',
                 alpha=alpha)

        # 3/2-
        ax1.plot(tof_x, tof_03, label=r'$3/2- \ n \alpha$', color='b',
                 alpha=alpha)

        # nn
        ax1.plot(tof_x, tof_nn, label=r'nn', color='g',
                 alpha=alpha)

        # Figure 2
        # --------
        # Total
        ax2.plot(tof_x, tof_dt + tof_tot + tof_sc, 'r-', label='total',
                 alpha=alpha)

        # TT total
        ax2.plot(tof_x, tof_tot, label='TT total', color='C1',
                 alpha=alpha)

        # 1/2̈́+
        ax2.plot(tof_x, tof_01, label=r'$1/2+ \ n \alpha$', color='c',
                 alpha=alpha)

        # interference
        ax2.plot(tof_x, tof_in, label='interference', color='k',
                 alpha=alpha)

        if save_specs:
            tof_specs = [tof_tot.tolist(), tof_01.tolist(), tof_02.tolist(), 
                         tof_03.tolist(), tof_nn.tolist(), tof_in.tolist()]
            E_specs = [tt_tot[:, 1].tolist(), tt_01[:, 1].tolist(), 
                       tt_02[:, 1].tolist(), tt_03[:, 1].tolist(), 
                       tt_nn[:, 1].tolist()]
            specs['tof specs'].append(tof_specs)
            specs['E specs'].append(E_specs)
            specs['feed params'].append(feed.tolist())
    
    legend_el = [Line2D([0], [0], color='r', label='total'),
                 Line2D([0], [0], color='C1', label='TT total'),
                 Line2D([0], [0], color='k', label=r'$1/2^- \ n \alpha$'),
                 Line2D([0], [0], color='b', label=r'$3/2^- \ n \alpha$'),
                 Line2D([0], [0], color='g', label='nn')]
    ax1.legend(handles=legend_el)
    ax1.set_xlabel('$t_{TOF}$ (ns)')
    ax1.set_ylabel('counts')

    legend_el = [Line2D([0], [0], color='r', label='total'),
                 Line2D([0], [0], color='C1', label='TT total'),
                 Line2D([0], [0], color='c', label=r'$1/2^+ \ n \alpha$'),
                 Line2D([0], [0], color='k', label='interference')]
    ax2.legend(handles=legend_el)
    ax2.set_xlabel('$t_{TOF}$ (ns)')
    ax2.set_ylabel('counts')

    if save_specs:
        specs['x_tof'] = tof_x.tolist()
        specs['x_E'] = tt_tot[:, 0].tolist()
        udfs.json_write_dictionary('specs.json', udfs.listify(specs))
        
        

if __name__ == '__main__':
    # Read MCMC file
    path = 'output_files/mcmc/nbi/mcmc_output.json'
    mcmc = udfs.numpify(udfs.json_read_dictionary(path))
    C_stat = -mcmc['test_stat']

    params = mcmc['feed']
    modifiers = mcmc['samples']
    C_stat = C_stat
#
#    # Plot C-stat
#    plot_test_stat(C_stat, 'C-stats', n=100)
#
#    # Plot feed factors
#    plot_feed_factors(params, n=100)
#
#    # Plot modifiers
#    plot_modifiers(modifiers, n=100)

    # Plot TOF for given feed factors
    tof_path = 'data/nbi.txt'
    drf_path = '/home/beriksso/NES/drf/26-11-2022/tofu_drf_scaled_kin_ly.json'
    rigid_shift = -0.7
    n = 50
    exe = 'fortran/run_fortran'
    plot_tof(exe, params[-n:], tof_path, drf_path, rigid_shift, True)
