#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:37:55 2023

@author: beriksso
"""

import numpy as np
import matplotlib.pyplot as plt
import useful_defs as udfs
udfs.set_nes_plot_style()
from nes.tofor.commands import load_response_function
import time
import subprocess


def generate_tt_spec(exe, args):
    """
    Call C++ wrapper to run Fortran R-matrix code to generate TT neutron
    spectrum which is written to a text file.

    Parameters
    ----------
    exe : str,
        String containing the path to the executable.
    args : list,
         List of strings containing feed factor parameter values.

    Returns
    -------
    spec : ndarray,
        Array with 6 columns. Contains the TT neutron energy spectrum with
        all the feed factors set to the values given by 'args'. The first
        column contains the energy axis, the second contains the total
        spectrum, the third column contains the primary spectrum, the fourth
        column contains the secondary spectrum, the fifth columns contains
        the exchange spectrum, the sixth column contains the interference
        spectrum.
        
    Notes
    -----
    See generate_components() for additional information.
    """
    if isinstance(args, np.ndarray):
        args = [str(a) for a in args]

    # Define the command to run
    command = [exe] + args

    t0 = time.time()
    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    # Wait for the command to complete
    stdout, stderr = process.communicate()

    # Print the errors (if any)
    print(stderr.decode())
    print(f'Time: {time.time() - t0} s')

    # Return the output
    spec = stdout.decode().split()
    try:
        spec = np.array(spec, dtype='float').reshape([-1, 6])
    except:
        print(spec)
        return spec

    spec[:, 0] = 1000 * spec[:, 0]  # keV

    return spec


def generate_components(exe, feed):
    """
    Generate the TT components for the given feed factors.

    Notes
    -----
    The total spectrum when considering each channel seperately (i.e. when we
    set only the two feed factors affecting the channel, and the rest to zero)
    is given by summing the primary, secondary and exchange contributions from
    each channel, and then adding the interference contribution from the total
    fit (i.e. when all feed factors are set).

    Examples
    --------
    For, e.g., the 3/2- n alpha spectrum we get the total spectrum from the
    first column. This is equal to summing the primary, secondary, and
    exchange contributions, i.e. tt_03[:, 1] is the same as tt_03[:, 2] +
    tt_03[:, 3] + tt_03[:, 4]. The interference contribution here
    (tt_03[:, 4]) is zero since only the 3/2- feed factors have non-zero
    values.

    If we want the total spectrum we can either get it from tt_tot[:, 1] or by
    summing the 1/2+, 1/2-, 3/2-, nn, and interference components, i.e.
    tt_01[:, 1] + tt_02[:, 1] + tt_03[:, 1] + tt_nn[:, 1] + tt_tot[:, 4]

    Returns
    -------
    tt_tot : ndarray,
           Array with 6 columns. Contains the TT neutron energy spectrum with
           all the feed factors set to the values given by 'feed'. The first
           column contains the energy axis, the second contains the total
           spectrum, the third column contains the primary spectrum, the fourth
           column contains the secondary spectrum, the fifth columns contains
           the exchange spectrum, the sixth column contains the interference
           spectrum.
    tt_01 : ndarray,
          Array with 6 columns. Same as tt_tot but for the case where the two
          feed factors for the 1/2- n alpha channel are non-zero and the rest
          are zero.
    tt_02 : ndarray,
          Array with 6 columns. Same as tt_tot but for the case where the two
          feed factors for the 1/2+ n alpha channel are non-zero and the rest
          are zero.
    tt_03 : ndarray,
          Array with 6 columns. Same as tt_tot but for the case where the two
          feed factors for the 3/2- n alpha channel are non-zero and the rest
          are zero.
    tt_nn : ndarray,
          Array with 6 columns. Same as tt_tot but for the case where the two
          feed factors for the nn dineutron emission channel are non-zero and
          the rest are zero.
    """

    if isinstance(feed, np.ndarray):
        feed = [str(f) for f in feed]

    # Generate the total spectrum
    tt_tot = generate_tt_spec(exe, feed)

    # Generate 1/2+ n alpha
    arg = len(feed) * ['0.0']
    arg[0] = feed[0]
    arg[1] = feed[1]
    tt_01 = generate_tt_spec(exe, arg)

    # Generate 1/2- n alpha
    arg = len(feed) * ['0.0']
    arg[2] = feed[2]
    arg[3] = feed[3]
    tt_02 = generate_tt_spec(exe, arg)

    # Generate 3/2- n alpha
    arg = len(feed) * ['0.0']
    arg[4] = feed[4]
    arg[5] = feed[5]
    tt_03 = generate_tt_spec(exe, arg)

    # Generate nn
    arg = len(feed) * ['0.0']
    arg[6] = feed[6]
    tt_nn = generate_tt_spec(exe, arg)

    return tt_tot, tt_01, tt_02, tt_03, tt_nn


def calculate_tt_tof(drf, rigid_shift, tt_x, tt_y):
    """
    Calculate the TT tof spectrum from TT neutron emission energy spectrum.

    Parameters
    ----------
    drf : data.response object,
        Object from nes.Data containing the detector response matrix.
    rigid_shift : float,
                Shift in ns to apply to x-axis.
    tt_x : ndarray,
         Horizontal energy (keV) axis of TT spectrum, 1D array.
    tt_y : ndarray,
         Vertical intensity (a.u.) axis of TT spectrum, 1D array.

    Returns
    -------
    tt_tof_y : ndarray,
             Vertical intensity (counts/bin) axis.
    """
    # Interpolate to DRF axis
    tt_yi = udfs.interpolate_new_axis(drf.from_axis, tt_x, tt_y)

    # Calculate TOF spectrum with rigid shift
    tt_tof_y = np.dot(drf.matrix.T, tt_yi)
    tt_tof_x = drf.to_axis + rigid_shift

    # Interpolate to DRF tof axis
    tt_tof_yi = udfs.interpolate_new_axis(drf.to_axis, tt_tof_x, tt_tof_y)

    return tt_tof_yi


def plot_components(tt_tot, tt_01, tt_02, tt_03, tt_nn):
    """Plot the TT spectrum components for given feed factors."""
    x_axis = tt_tot[:, 0]

    plt.figure('TT components 1')
    plt.plot(x_axis, tt_tot[:, 1], 'k-', label='total')
    plt.plot(x_axis, tt_02[:, 1], linestyle='dotted', color='r',
             label=r'1/2$^-$ n$\alpha$', linewidth=1.5)
    plt.plot(x_axis, tt_03[:, 1], linestyle='--', color='b',
             label=r'3/2$^-$ n$\alpha$')
    plt.plot(x_axis, tt_nn[:, 1], linestyle='-.', color='g', label='nn')
    plt.xlabel('$E_n$ (keV)')
    plt.ylabel('Relative intensity (a.u.)')
    plt.gca().ticklabel_format(style='plain')
    plt.legend()

    plt.figure('TT components 2')
    plt.plot(x_axis, tt_tot[:, 1], 'k-', label='total')
    plt.plot(x_axis, tt_01[:, 1], linestyle='dotted', color='r',
             linewidth=1.5, label=r'1/2$^+$ n$\alpha$')
    plt.plot(x_axis, tt_tot[:, 5], linestyle='--', color='b',
             label='interference')
    plt.xlabel('$E_n$ (keV)')
    plt.ylabel('Relative intensity (a.u.)')
    plt.gca().ticklabel_format(style='plain')
    plt.legend()


def plot_tof(exe, feed, tof_path, drf_path, rigid_shift):
    """Plot TOF spectrum on data for given feed factors."""
    # Create TT energy spectrum
    tt_tot, tt_01, tt_02, tt_03, tt_nn = generate_components(exe, feed)
    bt_td = udfs.json_read_dictionary('input_files/specs/bt_td_spec.json')
    scatter = udfs.json_read_dictionary('input_files/specs/scatter_spec.json')

    # Translate to TOF
    drf = load_response_function(drf_path)
    tof_tot = calculate_tt_tof(drf, rigid_shift, tt_tot[:, 0], tt_tot[:, 1])
    tof_01 = calculate_tt_tof(drf, rigid_shift, tt_01[:, 0], tt_01[:, 1])
    tof_02 = calculate_tt_tof(drf, rigid_shift, tt_02[:, 0], tt_02[:, 1])
    tof_03 = calculate_tt_tof(drf, rigid_shift, tt_03[:, 0], tt_03[:, 1])
    tof_nn = calculate_tt_tof(drf, rigid_shift, tt_nn[:, 0], tt_nn[:, 1])
    tof_in = calculate_tt_tof(drf, rigid_shift, tt_tot[:, 0], tt_tot[:, -1])
    tof_dt = bt_td['tof']['y']
    tof_sc = scatter['tof']['y']

    # Read TOF data
    tof_x, tof_y, tof_bgr = np.loadtxt(tof_path, delimiter=',', unpack=True)

    # Figure 1
    # --------
    plt.figure('Fig 1')
    # Data
    plt.errorbar(tof_x, tof_y - tof_bgr, yerr=np.sqrt(tof_y), linestyle='None',
                 color='k', marker='.', markersize=1)

    # Total fit
    plt.plot(tof_x, tof_dt + tof_tot + tof_sc, 'r-', label='total')

    # DT
    plt.plot(tof_x, tof_dt, 'C0-', marker='None', label='DT',
             linestyle=udfs.get_linestyle('dashed'))

    # TT
    plt.plot(tof_x, tof_tot, linestyle='--', marker='None', color='C1',
             label='TT total')

    # Scatter
    plt.plot(tof_x, tof_sc, 'k-.', label='scatter')
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('counts')
    plt.legend()

    # Figure 2
    # --------
    plt.figure('Fig 2')
    # Data
    plt.errorbar(tof_x, tof_y - tof_bgr, yerr=np.sqrt(tof_y), linestyle='None',
                 color='k', marker='.', markersize=1)

    # Total
    plt.plot(tof_x, tof_dt + tof_tot + tof_sc, 'r-', label='total')

    # TT total
    plt.plot(tof_x, tof_tot, label='TT total', color='C1',
             linestyle=udfs.get_linestyle('long dash with offset'))

    # 1/2̈́+
    if float(feed[0]) or float(feed[1]):
        plt.plot(tof_x, tof_01, linewidth=1.5, color='c',
                 label=r'$1/2+ \ n \alpha$',
                 linestyle=udfs.get_linestyle('dotted'))

    # 1/2-
    if float(feed[2]) or float(feed[3]):
        plt.plot(tof_x, tof_02, linewidth=1.5, color='r',
                 label=r'$1/2- \ n \alpha$',
                 linestyle=udfs.get_linestyle('densely dotted'))

    # 3/2-
    if float(feed[4]) or float(feed[5]):
        plt.plot(tof_x, tof_03, label=r'$3/2- \ n \alpha$',
                 linestyle=udfs.get_linestyle('dashed'), color='C0')

    # nn
    if float(feed[6]):
        plt.plot(tof_x, tof_nn, label=r'nn', color='g',
                 linestyle=udfs.get_linestyle('dashdotted'))

    # interference
    plt.plot(tof_x, tof_in, label='interference', color='k',
             linestyle=udfs.get_linestyle('densely dashdotted'))

    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('counts')

    plt.legend()


if __name__ == '__main__':
    feed = np.loadtxt('input_files/feed_pars/nbi/p0.txt', usecols=1)
    exe = 'fortran/run_fortran'
    
     # Generate TT components
    tt_comps = generate_components(exe, feed)
    
    # Plot
    plot_components(*tt_comps)

    # Plot TOF for given feed factors
    tof_path = 'data/nbi.txt'
    drf_path = '/home/beriksso/NES/drf/26-11-2022/tofu_drf_scaled_kin_ly.json'
    rigid_shift = -0.7
    plot_tof(exe, feed, tof_path, drf_path, rigid_shift)
