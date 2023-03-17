#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:50:34 2023

@author: beriksso
"""

import numpy as np
#from rmatspec import generate_tt_spec
import rmatspec as rms
import useful_defs as udfs
import nes
from nes import tofor
from nes.tofor.commands import load_response_function
import scipy as sp
import time
import os
import sys
import datetime
import matplotlib.pyplot as plt


def load(file_name, name=None, shot=None, t0=None, t1=None, drf=''):
    """Load TOFu data and TOFOR response function."""
    # Load data from file
    dat = np.loadtxt(file_name, delimiter=',')
    bins = dat[:, 0]
    counts = dat[:, 1]
    bgr = dat[:, 2]

    # Load response function
    r = load_response_function(drf)

    if name is None:
        name = file_name

    nes_data = nes.Data(counts, name=name, response=r, back=bgr,
                        x_label='$t_{TOF}$ (ns)', y_label='counts/bin')

    # JET specific attributes
    nes_data.shot = shot
    nes_data.t0 = t0
    nes_data.t1 = t1

    return nes_data


def load_tt(norm, path):
    """Return TT spectrum from input_files."""
    tt = np.loadtxt(path)
    tt_x = np.array(tt[:, 0]) * 1000
    tt_y = np.array(tt[:, 1])
    
    return tt_x, tt_y * norm


def get_normalization(spec_1, spec_2):
    """
    Return normalization constant to normalize integral under spec_1 to 
    integral under spec_2.
    """
    n1 = np.trapz(spec_1[1], x=spec_1[0])
    n2 = np.trapz(spec_2[1], x=spec_2[0])
    
    return n2/n1


def cash(observed, expected):
    """Return Cash statistic."""
    mask = (observed > 0) & (expected > 0)
    cash = (-2 * np.sum(observed[mask] * np.log(expected[mask]/observed[mask]) 
            + observed[mask] - expected[mask]))
    
    return cash

def chi2r(observed, expected):
    """Return reduced Chi2."""
    mask = (observed > 0) & (expected > 0)
    chi2r = (np.sum((observed[mask] - expected[mask])**2 / expected[mask]) 
            / (len(observed[mask]) - 3))

    return chi2r
    

def tofor_fit_function(fit_stat='cash'):
    """Fit function for fit_tofor()."""
    # TOF components
    dt_tof = tofor.fit.comp_data['D(T,n)He4']
    tt_tof = tofor.fit.comp_data['TT total']
    scatter_tof = tofor.fit.comp_data['scatter']
    tot = dt_tof + tt_tof + scatter_tof + tofor.fit.data.back
    
    # Mask for given fit range
    mask = ((tofor.fit.data.axis >= tofor.fit.data_xlim[0]) & 
            (tofor.fit.data.axis <= tofor.fit.data_xlim[1]))
    
    if fit_stat == 'cash':
        stat = cash(tofor.fit.data.data[mask], tot[mask])
    elif fit_stat == 'chi2r':
        stat = chi2r(tofor.fit.data.data[mask], tot[mask])
    
    return stat


def fit_tofor(fit_stat='cash'):
    """
    Replacement function for tofor.fit() which runs slow when submitted as 
    a batch job.
    """
    """Fit function for fit_tofor()."""
    # TOF components
    dt_tof = tofor.fit.comp_data['D(T,n)He4']
    tt_tof = tofor.fit.comp_data['TT total']
    scatter_tof = tofor.fit.comp_data['scatter']
    
    tot = dt_tof + tt_tof + scatter_tof + tofor.fit.data.back
    
    # Mask for given fit range
    mask = ((tofor.fit.data.axis >= tofor.fit.data_xlim[0]) & 
            (tofor.fit.data.axis <= tofor.fit.data_xlim[1]))
    
    if fit_stat == 'cash':
        stat = cash(tofor.fit.data.data[mask], tot[mask])
    elif fit_stat == 'chi2r':
        stat = chi2r(tofor.fit.data.data[mask], tot[mask])
    
    return stat


def plot_tof_spec(pars=(1, 1, 1)):
    """Plot fit performed by fit_function"""
    dt = pars[0] * tofor.fit.comp_data['D(T,n)He4']
    tt = pars[1] * tofor.fit.comp_data['TT total']
    scatter = pars[2] * tofor.fit.comp_data['scatter']
    
    plt.figure('TOF spec')
    plt.plot(tofor.fit.data.axis, dt)
    plt.plot(tofor.fit.data.axis, tt)
    plt.plot(tofor.fit.data.axis, scatter)
    plt.plot(tofor.fit.data.axis, dt + tt + scatter)
    plt.errorbar(tofor.fit.data.axis, tofor.fit.data.data, 
                 yerr=np.sqrt(tofor.fit.data.data + tofor.fit.data.back),
                 marker='.', markersize=1.0, color='k', linestyle='None')
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('counts')


def fit_function(feed, exe, norm_tt, norm_p0, out_file, temp_path, verbose=False):
    """Return C-stat for TT spectrum and data for given feed parameters."""
    t_tot = time.time()
    # Generate TT spectrum
    arguments = [str(f * n) for f, n in zip(feed, norm_p0)]
    t0 = time.time()
    tt_specs = rms.generate_tt_spec(exe, arguments)
    tt_x = tt_specs[:, 0]
    tt_y = tt_specs[:, 1]
    write_time('generate_tt_spec():', time.time() - t0, out_file)

    t0 = time.time()

    tt_tof_yi = rms.calculate_tt_tof(tofor.fit.data.response, 
                                     tofor.fit.rigid_shift.value, tt_x, tt_y)

    tofor.fit.comp_data['TT total'] = tt_tof_yi * norm_tt
    write_time('Set NES manually', time.time() - t0, out_file)
    
    t0 = time.time()
    C_stat = fit_tofor()
    write_time('fit_tofor():', time.time() - t0, out_file)
    
    t0 = time.time()
    if verbose:
        chi2r = tofor_fit_function('chi2r')
        print(f'fit_function() finished in: {time.time() - t0:.2f} s')
        print(f'C = {C_stat}')
        print(f'chi2r = {chi2r}')
        
        to_save = {'bt_td': [tofor.fix1.En.tolist(), 
                             tofor.fix1.shape.tolist(), 
                             (tofor.fit.comp_data['D(T,n)He4']).tolist()],
                   'tt': [tt_x.tolist(), 
                           tt_y.tolist(), 
                          (tofor.fit.comp_data['TT total']).tolist()],
                   'scatter': [tofor.scatter.En.tolist(), 
                               tofor.scatter.shape.tolist(), 
                               (tofor.fit.comp_data['scatter']).tolist()],
                   'feed': arguments,
                   'C_stat': C_stat,
                   'chi2r': chi2r,
                   'modifiers': feed.tolist(),
                   'p0': norm_p0.tolist()}
        if temp_path == './':
            f_name = 'output.json'
        else:
            files = os.listdir(temp_path)
            f_name = f'{temp_path}/fit_{len(files) + 1}.json'
        udfs.json_write_dictionary(f_name, to_save)
    write_time('Saving json:', time.time() - t0, out_file)
    
    write_time('fit_function tot:', time.time() - t_tot, out_file)
    return C_stat


def check_temp_files(path):
    """Remove all files from temporary directory."""
    files = os.listdir(path)
    if len(files) != 0:
        ans = input(f'Remove files from {path}? (y/n) ')
        if ans == 'y':
            for file in files:
                os.remove(f'{path}/{file}')


def write_time(text, time, out_file):
    """Write string to info.txt."""
    t = str(np.round(time, 2))
    with open(out_file, 'a') as handle:
        handle.write("{:<30}{}\n".format(text, t))


def set_components(out_file):
    """Set fit components."""
    # Set BT TD component
    bt_td = udfs.json_read_dictionary('input_files/specs/bt_td_spec.json')
    bt_td_comp = tofor.fix1
    bt_td_comp.En = np.array(bt_td['x']) # keV
    bt_td_comp.shape = np.array(bt_td['y'])
    bt_td_comp.N = bt_td['N']
    bt_td_comp.N.lock = False
    bt_td_comp.name = 'D(T,n)He4'
    bt_td_comp.use = True
    
    # Set scatter component
    scatter = udfs.json_read_dictionary('input_files/specs/scatter_spec.json')
    tofor.scatter.En = np.array(scatter['x']) # keV
    tofor.scatter.shape = np.array(scatter['y'])
    tofor.scatter.N = scatter['N']
    tofor.scatter.N.lock = False
    tofor.scatter.use = True
    
    # Set TT component
    tt_x, tt_y = load_tt(1, 'input_files/specs/tt_spec.txt')
    norm_tt = get_normalization((tt_x, tt_y), (bt_td_comp.En, bt_td_comp.shape))
    tt_comp = tofor.fix2
    tt_comp.En = tt_x
    tt_comp.shape = tt_y * norm_tt
    tt_comp.N = bt_td_comp.N.value
    tt_comp.name = 'TT total'
    tt_comp.use = True
    
    # Allow rigid shift
    tofor.fit.rigid_shift = -0.7
    tofor.fit.rigid_shift.lock = True
    tofor.fit.rigid_shift.max = 1.5
    tofor.fit.rigid_shift.min = -1.5
    tofor.fit.plot_model = True
    
    t0 = time.time()
    tofor.fit()
    write_time('First fit:', time.time() - t0, out_file)
        
    return norm_tt


def main(start_params, temp_path, out_file, dat_file, verbose=False):
    check_temp_files(temp_path)
    now = datetime.datetime.now()
    print(now)
    with open(out_file, 'w') as handle:
        handle.write(f'Fitting procedure started at {now}\n')
    
    # Set NES components
    # ------------------
    
    # Decide which DRF to use (generated using light yield or not)
    light_yield = True
    suffix = '_ly' if light_yield else ''
    
    # Load DRF and data
    drf = f'/home/beriksso/NES/drf/26-11-2022/tofu_drf_scaled_kin{suffix}.json'
    data = load(dat_file, drf=drf, name='')
    tofor.fit.data = data
    tofor.fit.data_xlim = (20, 80)
    
    # Set fit components
    norm_tt = set_components(out_file)
    
    # Start feed parameter fitting procedure
    # --------------------------------------
    tofor.fit.data_xlim = (32.5, 80)
    
    # Read start values for feed parameters
    feed = np.loadtxt(start_params, dtype='str')
    norm_p0 = np.array(feed[:, 1], dtype='float')
    
    # Use parameters normalized to 1
    p0 = np.ones(len(norm_p0))
    
    """
    For some reason this doesn't return the minimum C_stat. Set verbose=True and
    find the true minimum from all the output files by running plot_fits.py.
    """
    exe = 'fortran/run_fortran'
    popt = sp.optimize.minimize(fit_function, p0, options={'eps': 0.025},
                                args=(exe, norm_tt, norm_p0, out_file, 
                                      temp_path, verbose))

    return popt


if __name__ == '__main__':
    out = 'temp_01'
    start_params = 'input_files/feed_pars/p0_16.txt'
    temp_path = f'/common/scratch/beriksso/TOFu/data/r_matrix/fit_files/{out}'
    out_file = f'{out}.txt'
    dat_file = 'data/nbi.txt'
    verbose = True
    popt = main(start_params, temp_path, out_file, dat_file, verbose)

