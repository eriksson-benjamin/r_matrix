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


def fit_function(feed, exe, norm_tt, norm_p0, out_file, temp_path, verbose):
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
    C_stat = return_fit_stat('cash')
    write_time('fit_tofor():', time.time() - t0, out_file)

    t0 = time.time()
    if verbose:
        chi2r = return_fit_stat('chi2r')
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


def set_components(out_file):
    """Set fit components."""
    # Set BT TD component
    bt_td = udfs.json_read_dictionary('input_files/specs/bt_td_spec.json')
    bt_td_comp = tofor.fix1
    bt_td_comp.En = np.array(bt_td['x'])  # keV
    bt_td_comp.shape = np.array(bt_td['y'])
    bt_td_comp.N = bt_td['N']
    bt_td_comp.N.lock = False
    bt_td_comp.name = 'D(T,n)He4'
    bt_td_comp.use = True

    # Set scatter component
    scatter = udfs.json_read_dictionary('input_files/specs/scatter_spec.json')
    tofor.scatter.En = np.array(scatter['x'])  # keV
    tofor.scatter.shape = np.array(scatter['y'])
    tofor.scatter.N = scatter['N']
    tofor.scatter.N.lock = False
    tofor.scatter.use = True

    # Set TT component
    tt_x, tt_y = load_tt(1, 'input_files/specs/tt_spec.txt')
    norm_tt = get_normalization(
        (tt_x, tt_y), (bt_td_comp.En, bt_td_comp.shape))
    tt_comp = tofor.fix2
    tt_comp.En = tt_x
    tt_comp.shape = tt_y * norm_tt
    tt_comp.N = bt_td_comp.N.value
    tt_comp.name = 'TT total'
    tt_comp.use = True

    # Rigid shift
    tofor.fit.rigid_shift = -0.7
    tofor.fit.rigid_shift.lock = True
    tofor.fit.plot_model = True

    t0 = time.time()
    tofor.fit()
    write_time('First fit:', time.time() - t0, out_file)

    return norm_tt


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

    return n2 / n1


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


def return_fit_stat(fit_stat='cash'):
    """Return fit statistic."""
    # TOF components
    dt_tof = tofor.fit.comp_data['D(T,n)He4']
    tt_tof = tofor.fit.comp_data['TT total']
    scatter_tof = tofor.fit.comp_data['scatter']

    tot = dt_tof + tt_tof + scatter_tof + tofor.fit.data.back

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


