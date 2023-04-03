#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:16:19 2023

@author: beriksso
"""

import useful_defs as udfs
import time
import rmatspec as rms
import numpy as np
import os
from nes import tofor
import datetime

def fit_function(feed, exe, norm_p0, components, out_file, temp_path, verbose):
    """Return C-stat for TT spectrum and data for given feed parameters."""
    t_tot = time.time()
    # Generate TT energy spectrum
    arguments = [str(f * n) for f, n in zip(feed, norm_p0)]
    t0 = time.time()
    tt_specs = rms.generate_tt_spec(exe, arguments)
    tt_x = tt_specs[:, 0]
    tt_y = tt_specs[:, 1]
    write_time('generate_tt_spec():', time.time() - t0, out_file)

    t0 = time.time()
    # Translate to time-of-flight
    tt_tof_yi = rms.calculate_tt_tof(tofor.fit.data.response,
                                     tofor.fit.rigid_shift.value, tt_x, tt_y)
    components['TT total'] = tt_tof_yi
    write_time('Set NES manually', time.time() - t0, out_file)

    t0 = time.time()
    # Calculate Cash-statistic
    C_stat = return_fit_stat(components, 'cash')
    write_time('fit_tofor():', time.time() - t0, out_file)

    t0 = time.time()
    
    if verbose:
        verbose_function(components, tt_x, tt_y, arguments, temp_path, 
                         C_stat, feed, norm_p0)
    
    write_time('Saving json:', time.time() - t0, out_file)

    write_time('fit_function tot:', time.time() - t_tot, out_file)
    return C_stat


def verbose_function(components, tt_x, tt_y, arguments, temp_path, 
                     C_stat, feed, norm_p0):
    chi2r = return_fit_stat(components, 'chi2r')
    
    print(f'C = {C_stat}')
    print(f'chi2r = {chi2r}')
    
    
    to_save = {'bt_td': components['D(T,n)He4'].tolist(),
               'tt': [tt_x.tolist(),
                      tt_y.tolist(),
                      (components['TT total']).tolist()],
               'scatter': components['scatter'].tolist(),
               'feed': arguments,
               'C_stat': C_stat,
               'chi2r': chi2r,
               'modifiers': feed.tolist(),
               'p0': norm_p0.tolist()}
    if temp_path == './':
        f_name = 'output.json'
    else:
        # Count fit files
        files = os.listdir(temp_path)
        f_count = [file for file in files if file.split('_')[0] == 'fit' ]
        
        # Set file name
        f_name = f'{temp_path}/fit_{len(f_count) + 1}.json'
    
    now = datetime.datetime.now()
    to_save['datetime'] = now.strftime('%Y-%m-%d %H:%M:%S')
    udfs.json_write_dictionary(f_name, to_save)


def set_components(dt_spec, scatter_spec, mi_spec, rigid_shift=-0.7):
    """
    Set fit components.
    """
    components = {}

    # Set BT TD component (time-of-flight)
    bt_td = udfs.json_read_dictionary(dt_spec)
    components['D(T,n)He4'] = np.array(bt_td['tof']['y'])

    # Set scatter component (time-of-flight)
    scatter = udfs.json_read_dictionary(scatter_spec)
    components['scatter'] = np.array(scatter['tof']['y'])

    # Rigid shift
    tofor.fit.rigid_shift = rigid_shift

    return components


def load_tt(path):
    """Return TT spectrum from input_files."""
    tt = np.loadtxt(path)
    tt_x = np.array(tt[:, 0]) * 1000
    tt_y = np.array(tt[:, 1])

    return tt_x, tt_y


def cash(observed, expected):
    """Return Cash statistic."""
    mask = (observed > 0) & (expected > 0)
    cash = (-2*np.sum(observed[mask] * np.log(expected[mask]/observed[mask]) 
                      + observed[mask] - expected[mask]))

    return cash


def chi2r(observed, expected):
    """Return reduced Chi2."""
    mask = (observed > 0) & (expected > 0)
    chi2r = (np.sum((observed[mask] - expected[mask])**2 / expected[mask])
             / (len(observed[mask]) - 3))

    return chi2r


def return_fit_stat(components, fit_stat='cash'):
    """Return fit statistic."""
    #Sum TOF components
    tot = np.zeros(len(tofor.fit.data.data))
    for val in components.values():
        tot += val

    # Add background component from data
    tot += tofor.fit.data.back

    # Mask for given fit range
    mask = ((tofor.fit.data.axis >= tofor.fit.data_xlim[0])
            & (tofor.fit.data.axis <= tofor.fit.data_xlim[1]))

    if fit_stat == 'cash':
        stat = cash(tofor.fit.data.data[mask], tot[mask])
    elif fit_stat == 'chi2r':
        stat = chi2r(tofor.fit.data.data[mask], tot[mask])

    return stat


def write_time(text, time, out_file):
    """Write string to info.txt."""
    t = str(np.round(time, 2))
    with open(out_file, 'a') as handle:
        handle.write("{:<30}{}\n".format(text, t))


def check_temp_files(path):
    """Remove all files from temporary directory."""
    files = os.listdir(path)
    if len(files) != 0:
        ans = input(f'Remove files from {path}? (y/n) ')
        if ans == 'y':
            for file in files:
                os.remove(f'{path}/{file}')


