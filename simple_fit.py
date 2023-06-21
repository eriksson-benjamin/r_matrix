#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:50:34 2023

@author: beriksso
"""

import numpy as np
import rmatfit as rmf
import useful_defs as udfs
import nes
from nes import tofor
from nes.tofor.commands import load_response_function
import scipy as sp
import sys
import datetime
import matplotlib.pyplot as plt


def load(file_name, name=None, shot=None, t0=None, t1=None, drf=''):
    """Load TOFu data and TOFOR response function."""
    # Load data from file
    dat = np.loadtxt(file_name, delimiter=',')
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


def plot_tof_spec(pars=(1, 1, 1)):
    """Plot fit performed by fit_function."""
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



def main(start_params, temp_path, specs, out_file, dat_file, verbose=False):
    rmf.check_temp_files(temp_path)
    now = datetime.datetime.now()
    print(now)
    with open(out_file, 'w') as handle:
        handle.write(f'Fitting procedure started at {now}\n')

    # Decide which DRF to use (generated using light yield or not)
    light_yield = True
    suffix = '_ly' if light_yield else ''

    # Load DRF and data
    drf = f'/home/beriksso/NES/drf/26-11-2022/tofu_drf_scaled_kin{suffix}.json'
    data = load(dat_file, drf=drf, name='')
    tofor.fit.data = data
    tofor.fit.data_xlim = (20, 80)

    # Set fit components
    components = rmf.set_components(specs[0], specs[1], -0.7)

    # Start feed parameter fitting procedure
    # --------------------------------------
    tofor.fit.data_xlim = (32.5, 80)

    # Read start values for feed parameters
    feed = np.loadtxt(start_params, dtype='str')
    norm_p0 = np.array(feed[:, 1], dtype='float')

    # Use parameters normalized to 1
    p0 = np.ones(len(norm_p0))

    """
    For some reason this doesn't return the minimum C_stat. Set verbose=True
    and find the true minimum from all the output files by
    running plot_fits.py.
    """
    exe = 'fortran/run_fortran'
    popt = sp.optimize.minimize(rmf.fit_function, p0, options={'eps': 0.025},
                                args=(exe, norm_p0, components, out_file,
                                      temp_path, verbose))

    return popt


if __name__ == '__main__':
    out = 'temp_01'
    name = 'nbi'
    start_params = f'input_files/feed_pars/{name}/p0.txt'
    temp_path = f'/common/scratch/beriksso/TOFu/data/r_matrix/fit_files/{out}'
    out_file = f'{out}.txt'
    dat_file = f'data/{name}.txt'
    specs = [f'input_files/specs/{name}/bt_td_spec.json', 
             f'input_files/specs/{name}/scatter_spec.json']
    verbose = True
    popt = main(start_params, temp_path, specs, out_file, dat_file, verbose)
