#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:22:16 2023

@author: beriksso
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:46:10 2022

@author: beriksso
"""


import emcee
import matplotlib.pyplot as plt
import numpy as np
from nes import tofor
import useful_defs as udfs
import time
from datetime import datetime
import nes
from nes.tofor.commands import load_response_function
import sys
import rmatfit as rmf
import os
import shutil


# Global logging variable
log_level = (1, 'log_mcmc.txt')
if log_level[0]:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(log_level[1], 'w') as handle:
        handle.write(f'{dt_string} fit_tt_mcmc.py initiated.\n')


def print_log(log_file, log_msg):
    """Print log message."""
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(log_file, 'a') as handle:
        handle.write(f'{dt_string} {log_msg}\n')


def load(file_name, name=None, shot=None, t0=None, t1=None, drf=''):
    """Load TOF data and response matrix."""
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


def lnlike(parameters):
    """Return value of test statistic (C-stat)."""
    if log_level[0]:
        t_start = time.time()

    cash = rmf.fit_function(parameters, exe, norm_p0, out_file,
                            temp_path, verbose)

    print(f'C_stat = {cash:.2f}')
    if log_level[0]:
        msg = f'lnlike() finished in {time.time() - t_start} s.\n'
        print_log(log_level[1], msg)

    # Maximize this value (therefore negative sign)
    return -cash


def lnprior(parameters):
    """
    Define priors if required. No priors used currently.
    
    Notes
    -----
    Return 0 if parameters within bounds, else return -infinity.
    """

    return 0.0


def lnprob(parameters):
    """Return test statistic if parameters within bounds."""
    if log_level[0]:
        t_start = time.time()

    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf

    if log_level[0]:
        msg = f'lnprob() finished in {time.time() - t_start} s.'
        print_log(log_level[1], msg)

    # If lp is not -inf, its 0, so this just returns likelihood
    return lp + lnlike(parameters)


def main(p0, nwalkers, niter, ndim, lnprob, data):
    """Run MCMC sampler."""
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    print("Running burn-in...")
    if log_level[0]:
        print_log(log_level[1], 'Running burn-in.')
    p0, _, _ = sampler.run_mcmc(p0, 40, progress=True)
    sampler.reset()

    # Move burn in output files
    files = os.listdir(temp_path)
    for file in files:
        shutil.move(f'{temp_path}/{file}', f'{burn_in}/{file}')

    print("Running production...")
    if log_level[0]:
        print_log(log_level[1], 'Running production')
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state


def load_tt():
    """Return TT spectrum from output_files."""
    tt = np.loadtxt('input_files/tt_spec.txt')
    tt_x = np.array(tt[:, 0]) * 1000
    tt_y = np.array(tt[:, 1])

    return tt_x, tt_y


if log_level[0]:
    print_log(log_level[1], 'MCMC fitting procedure initiated.')

# Paths etc.
name = 'nbi'
file_name = f'data/{name}.txt'
out = 'mcmc_05'
out_file = f'{out}.txt'
temp_path = f'/common/scratch/beriksso/TOFu/data/r_matrix/fit_files/{out}'
burn_in = f'/common/scratch/beriksso/TOFu/data/r_matrix/fit_files/bi_{out}'
if not os.path.exists(burn_in):
    raise Exception('Burn in directory missing.')

# Remove old files
rmf.check_temp_files(temp_path)
rmf.check_temp_files(burn_in)

exe = 'fortran/run_fortran'
verbose = True

# Parameter files
param_file = 'p0_16'
start_params = f'input_files/feed_pars/{param_file}.txt'

# Read start values for feed parameters
feed = np.loadtxt(start_params, dtype='str')
norm_p0 = np.array(feed[:, 1], dtype='float')


# Decide which DRF to use (generated using light yield or not)
light_yield = True
suffix = '_ly' if light_yield else ''

# Load DRF and data
drf = f'/home/beriksso/NES/drf/26-11-2022/tofu_drf_scaled_kin{suffix}.json'
data = load(file_name, drf=drf, name='')
tofor.fit.data = data
tofor.fit.data_xlim = (20, 100)

rmf.set_components(out_file)

tofor.fit.data_xlim = (32.5, 80)

# Start feed parameter fitting procedure
# --------------------------------------

# Estimate model uncertainties using MCMC
dat = (data.axis, data.data, np.sqrt(data.data))

# Set number of walkers and number of iterations
n_walkers = 14
n_iter = 10000

# Normalized initial guess using p0 as normalization constants
initial = np.ones(len(norm_p0))
n_dim = len(initial)
p0 = [np.array(initial)
      + 1E-7 * np.random.randn(n_dim) for i in range(n_walkers)]
sampler, pos, prob, state = main(p0, n_walkers, n_iter, n_dim, lnprob, dat)

if log_level[0]:
    msg = 'MCMC fitting procedure finished. Saving to file...'
    print_log(log_level[1], msg)

# Save to file
to_save = {'samples': sampler.flatchain,
           'test_stat': sampler.flatlnprobability,
           'p0': norm_p0,
           'feed': sampler.flatchain * norm_p0}
udfs.json_write_dictionary(f'{temp_path}/mcmc_output.json',
                           udfs.listify(to_save))
