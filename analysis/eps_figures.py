#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:54:48 2023

@author: beriksso
"""

"""
Plot figures for the EPS conference 2023.
"""

import sys
sys.path.insert(0, '../')
import rmatspec as rms
import numpy as np
import matplotlib.pyplot as plt
import useful_defs as udfs
udfs.set_nes_plot_style()
from matplotlib.patches import FancyArrowPatch


def plot_components(tt_tot, tt_01, tt_02, tt_03, tt_nn):
    """Plot the TT spectrum components for given feed factors."""
    x_axis = tt_tot[:, 0] / 1000

    plt.figure('TT components 1')
    ax_1 = plt.gca()
    plt.plot(x_axis, tt_tot[:, 1], color='C1', label='total',
                 linestyle=udfs.get_linestyle('long dash with offset'))
    plt.plot(x_axis, tt_02[:, 1], label=r'1/2$^-$ n$\alpha$', linewidth=1.5,
             linestyle=udfs.get_linestyle('densely dotted'), color='r')
    plt.plot(x_axis, tt_03[:, 1], linestyle=udfs.get_linestyle('dashed'), 
             color='C0', label=r'3/2$^-$ n$\alpha$')
    plt.plot(x_axis, tt_nn[:, 1], color='g', label='nn',
             linestyle=udfs.get_linestyle('dashdotted'))
    plt.xlabel('$E_n$ (MeV)')
    plt.ylabel('Relative intensity (a.u.)')
    plt.gca().ticklabel_format(style='plain')
    plt.legend(loc='upper left')
    plt.xlim(-0.0, 10.000)
    plt.ylim(0, 65)
    ax_1.text(x=0.93, y=0.91, s='(a)', fontsize=12, transform=ax_1.transAxes)    
    plt.figure('TT components 2')
    ax_2 = plt.gca()
    plt.plot(x_axis, tt_tot[:, 1], color='C1', label='total',
             linestyle=udfs.get_linestyle('long dash with offset'))
    plt.plot(x_axis, tt_01[:, 1], color='c', label=r'1/2$^+$ n$\alpha$',
             linestyle=udfs.get_linestyle('dotted'), linewidth=1.5)
             
    plt.plot(x_axis, tt_tot[:, 5], color='k', label='interference',
             linestyle=udfs.get_linestyle('densely dashdotted'))
    plt.xlabel('$E_n$ (MeV)')
    plt.ylabel('Relative intensity (a.u.)')
    plt.gca().ticklabel_format(style='plain')
    plt.legend(loc='upper left')
    plt.xlim(0.0, 10.000)
    plt.ylim(-10, 65)
    ax_2.text(x=0.93, y=0.91, s='(b)', fontsize=12, transform=ax_2.transAxes)    
    return ax_1, ax_2


def plot_tof(exe, feed, tof_path, drf_path, name, rigid_shift):
    """Plot TOF spectrum on data for given feed factors."""
    factor = 1.05
    # Create TT energy spectrum
    tt_09 = rms.generate_tt_spec(exe, feed[0])
    tt_16 = rms.generate_tt_spec(exe, feed[1])
    bt_td = udfs.json_read_dictionary(f'../input_files/specs/{name}/bt_td_spec.json')
    scatter = udfs.json_read_dictionary(f'../input_files/specs/{name}/scatter_spec.json')

    # Translate to TOF
    drf = rms.load_response_function(drf_path)
    tof_09 = rms.calculate_tt_tof(drf, rigid_shift, tt_09[:, 0], tt_09[:, 1])
    tof_16 = rms.calculate_tt_tof(drf, rigid_shift, tt_16[:, 0], tt_16[:, 1])
    
    tof_dt = bt_td['tof']['y']
    tof_sc = scatter['tof']['y']

    # Read TOF data
    tof_x, tof_y, tof_bgr = np.loadtxt(tof_path, delimiter=',', unpack=True)

    tof_tots = (tof_09, tof_16)
    fit_nums = ('09', '16')
    titles = ('(b)', '(c)')
    
    for i, f in enumerate(feed):
        tt_tot, tt_01, tt_02, tt_03, tt_nn = rms.generate_components(exe, f)
        tof_01 = factor * rms.calculate_tt_tof(drf, rigid_shift, tt_01[:, 0], tt_01[:, 1])
        tof_02 = factor * rms.calculate_tt_tof(drf, rigid_shift, tt_02[:, 0], tt_02[:, 1])
        tof_03 = factor * rms. calculate_tt_tof(drf, rigid_shift, tt_03[:, 0], tt_03[:, 1])
        tof_nn = factor * rms.calculate_tt_tof(drf, rigid_shift, tt_nn[:, 0], tt_nn[:, 1])
        tof_in = factor * rms.calculate_tt_tof(drf, rigid_shift, tt_tot[:, 0], tt_tot[:, -1])
        
        # fit 09, fit 16
        # --------------
        plt.figure()
        # Data
        plt.errorbar(tof_x, tof_y - tof_bgr, yerr=np.sqrt(tof_y), linestyle='None',
                     color='k', marker='.', markersize=1)
    
        # Total
        plt.plot(tof_x, tof_dt + factor*tof_tots[i] + tof_sc, 'r-', label='total')
    
        # TT total
        plt.plot(tof_x, factor*tof_tots[i], label='TT total', color='C1',
                 linestyle=udfs.get_linestyle('long dash with offset'))
    
        # 1/2-
        if float(f[2]) or float(f[3]):
            plt.plot(tof_x, tof_02, linewidth=1.5, color='r',
                     label=r'$1/2^- \ n \alpha$',
                     linestyle=udfs.get_linestyle('densely dotted'))
    
        # 3/2-
        if float(f[4]) or float(f[5]):
            plt.plot(tof_x, tof_03, label=r'$3/2^- \ n \alpha$',
                     linestyle=udfs.get_linestyle('dashed'), color='C0')

        # 1/2̈́+
        if float(f[0]) or float(f[1]):
            plt.plot(tof_x, tof_01, linewidth=1.5, color='c',
                     label=r'$1/2^+ \ n \alpha$',
                     linestyle=udfs.get_linestyle('dotted'))
    
        # nn
        if float(f[6]):
            plt.plot(tof_x, tof_nn, label=r'nn', color='g',
                     linestyle=udfs.get_linestyle('dashdotted'))
    
        # interference
        plt.plot(tof_x, tof_in, label='interference', color='k',
                 linestyle=udfs.get_linestyle('densely dashdotted'))
    
        plt.xlabel('$t_{TOF}$ (ns)')
        plt.ylabel('counts')
        plt.legend()
        plt.xlim(30, 100)
        plt.ylim(-600, 4500)
        plt.title(f'Fit {fit_nums[i]}', loc='left', fontsize=12)
        plt.title(titles[i], loc='right', fontsize=12)
    


def plot_subplots(exe, feed, tof_path, drf_path, name, rigid_shift):
    """Plot TOF spectrum on data for given feed factors."""
    factor = 1.05
    # Create TT energy spectrum
    tt_09 = rms.generate_tt_spec(exe, feed[0])
    tt_16 = rms.generate_tt_spec(exe, feed[1])
    bt_td = udfs.json_read_dictionary(f'../input_files/specs/{name}/bt_td_spec.json')
    scatter = udfs.json_read_dictionary(f'../input_files/specs/{name}/scatter_spec.json')

    # Translate to TOF
    drf = rms.load_response_function(drf_path)
    tof_09 = rms.calculate_tt_tof(drf, rigid_shift, tt_09[:, 0], tt_09[:, 1])
    tof_16 = rms.calculate_tt_tof(drf, rigid_shift, tt_16[:, 0], tt_16[:, 1])
    
    tof_dt = bt_td['tof']['y']
    tof_sc = scatter['tof']['y']

    # Read TOF data
    tof_x, tof_y, tof_bgr = np.loadtxt(tof_path, delimiter=',', unpack=True)

    # Figure 1
    # --------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.subplots_adjust(wspace=0.05)
    tof_tots = (tof_09, tof_16)
    fit_nums = ('09', '16')
    titles = ('(b)', '(c)')
    
    font_size = 17
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
        ax.tick_params(axis='both', which='minor', labelsize=font_size - 2)
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)
    
    for i, f in enumerate(feed):
        ax = axes[i]

        tt_tot, tt_01, tt_02, tt_03, tt_nn = rms.generate_components(exe, f)
        tof_01 = factor * rms.calculate_tt_tof(drf, rigid_shift, tt_01[:, 0], tt_01[:, 1])
        tof_02 = factor * rms.calculate_tt_tof(drf, rigid_shift, tt_02[:, 0], tt_02[:, 1])
        tof_03 = factor * rms. calculate_tt_tof(drf, rigid_shift, tt_03[:, 0], tt_03[:, 1])
        tof_nn = factor * rms.calculate_tt_tof(drf, rigid_shift, tt_nn[:, 0], tt_nn[:, 1])
        tof_in = factor * rms.calculate_tt_tof(drf, rigid_shift, tt_tot[:, 0], tt_tot[:, -1])
        
        # Data
        ax.errorbar(tof_x, tof_y - tof_bgr, yerr=np.sqrt(tof_y), linestyle='None',
                     color='k', marker='.', markersize=1)
    
        # Total
        ax.plot(tof_x, tof_dt + factor*tof_tots[i] + tof_sc, 'r-', label='total')
    
        # TT total
        ax.plot(tof_x, factor*tof_tots[i], label='TT total', color='C1',
                 linestyle=udfs.get_linestyle('long dash with offset'))
    
        # 1/2-
        if float(f[2]) or float(f[3]):
            ax.plot(tof_x, tof_02, linewidth=1.5, color='r',
                     label=r'$1/2^- \ n \alpha$',
                     linestyle=udfs.get_linestyle('densely dotted'))
    
        # 3/2-
        if float(f[4]) or float(f[5]):
            ax.plot(tof_x, tof_03, label=r'$3/2^- \ n \alpha$',
                     linestyle=udfs.get_linestyle('dashed'), color='C0')

        # 1/2̈́+
        if float(f[0]) or float(f[1]):
            ax.plot(tof_x, tof_01, linewidth=1.5, color='c',
                     label=r'$1/2^+ \ n \alpha$',
                     linestyle=udfs.get_linestyle('dotted'))
    
        # nn
        if float(f[6]):
            ax.plot(tof_x, tof_nn, label=r'nn', color='g',
                     linestyle=udfs.get_linestyle('dashdotted'))
    
        # interference
        ax.plot(tof_x, tof_in, label='interference', color='k',
                 linestyle=udfs.get_linestyle('densely dashdotted'))
    
        ax.set_xlabel('$t_{TOF}$ (ns)', fontsize=font_size)
        ax.legend(fontsize=font_size-2)
        ax.set_xlim(30, 99)
        ax.set_ylim(-600, 4500)
        ax.set_title(f'Fit {fit_nums[i]}', fontsize=font_size,  loc='left')
        ax.set_title(titles[i], fontsize=font_size)
    return ax


def plot_tofu(tof_path, name):
    """Plot TOF spectrum on data for given feed factors."""
    # Read DT + scatter    
    bt_td = udfs.json_read_dictionary(f'../input_files/specs/{name}/bt_td_spec.json')
    scatter = udfs.json_read_dictionary(f'../input_files/specs/{name}/scatter_spec.json')
    tof_dt = bt_td['tof']['y']
    tof_sc = scatter['tof']['y']

    # Read TOF data
    tof_x, tof_y, tof_bgr = np.loadtxt(tof_path, delimiter=',', unpack=True)

    # Figure
    # ------
    plt.figure('TOF spectrum', figsize=(5, 5))
    ax = plt.gca()
    
    # Increase font size
    font_size = 17
    ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
    ax.tick_params(axis='both', which='minor', labelsize=font_size - 2)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    
    # Data
    plt.errorbar(tof_x, tof_y - tof_bgr, yerr=np.sqrt(tof_y), linestyle='None',
                 color='k', marker='.', markersize=1)

    # DT
    plt.plot(tof_x, tof_dt, 'C0-', marker='None', label='DT')

    # Line with arrowheads
    p1 = (32.5, 4600)
    p2 = (95.0, 4600)
    arrow = FancyArrowPatch(p1, p2, arrowstyle='<->', color='black', 
                            mutation_scale=20)
    ax.add_patch(arrow)
    ax.text((p1[0] + p2[0]) / 2, p1[1] + 400, 'T + T', ha='center', va='center', 
            fontsize=font_size - 2)
    # Scatter
    plt.plot(tof_x, tof_sc, 'k--', label='scatter')
    
    plt.xlabel('$t_{TOF}$ (ns)', fontsize=font_size)
    plt.ylabel('counts', fontsize=font_size)
    plt.legend(fontsize=font_size - 2)
    plt.xlim(20, 100)
    plt.ylim(0, 12000)
    plt.title(f'(a)', fontsize=font_size)
    plt.gca().ticklabel_format(style='plain')
    return ax

if __name__ == '__main__':
    # Read input feed factors
    p09 = np.loadtxt('input_files/feed_pars/p0_09.txt', usecols=1)
    p16 = np.loadtxt('input_files/feed_pars/p0_16.txt', usecols=1)
    
    # Generate components
    exe = '../fortran/run_fortran'
    
    feed = np.copy(p16) / 1E2
    tt_tot, tt_01, tt_02, tt_03, tt_nn = rms.generate_components(exe, feed)
    
    # Plot
    ax1, ax2 = plot_components(tt_tot, tt_01, tt_02, tt_03, tt_nn)
    
    # Plot TOF for given feed factors
    name = 'nbi'
    tof_path = f'../data/{name}/{name}.txt'
    drf_path = '/home/beriksso/NES/drf/26-11-2022/tofu_drf_scaled_kin_ly.json'
    rigid_shift = -0.7
    plot_tof(exe, (p09, p16), tof_path, drf_path, name, rigid_shift)

    plot_tofu(tof_path, name)
    plot_subplots(exe, (p09, p16), tof_path, drf_path, name, rigid_shift)
