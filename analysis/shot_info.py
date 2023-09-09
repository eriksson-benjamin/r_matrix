#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 08:22:54 2023

@author: beriksso
"""

"""
Read the list of shots and plot the plasma current, toroidal B-field, and the 
NBI power.
"""

import numpy as np
import matplotlib.pyplot as plt
import useful_defs as udfs
import find_cm_groups as fcg

if __name__ == '__main__':
    # Data set name
    name = 'nbi'
    
    # Read shot list
    shots, t0s, t1s = np.loadtxt(f'../data/{name}/shots_{name}.txt', unpack=True)
    shots = np.array(shots, dtype='int')
    current = np.array([])        
    B_field = np.array([])
    nbi_pow = np.array([])
    e_temp = np.array([])
    e_density = np.array([])
    counter = 0
    for shot, t0, t1 in zip(shots, t0s, t1s):
        # Import plasma current (MA)
        Ip, time = udfs.get_Ip(shot, (t0, t1))
        current = np.append(current,  Ip.mean() / 1E6)
    
        # Import toroidal B-field (T)
        Bt, time = udfs.get_Bt(shot, (t0, t1))
        B_field = np.append(B_field, Bt.mean())
        
        # Import NBI power (MW)
        P, time = udfs.get_NBI_power(shot, (t0, t1))
        nbi_pow = np.append(nbi_pow, P.mean() / 1E6)
    
        # Import the electron temperature (keV)
        Te = udfs.get_Te(shot, (t0, t1), diag='HRTX')
        e_temp = np.append(e_temp, np.mean(Te['HRTX'][1]))
    
        # Import the electron density (1/m^3)
        ne = udfs.get_ne(shot, (t0, t1), diag='HRTX')
        e_density = np.append(e_density, np.mean(ne['HRTX'][1]))
    
        print(f'{counter + 1}/{len(shots)} done.')
        counter += 1
    
    #%%
    # Plot weighted and non-weighted distributions of current/B-field/NBI power
    weights_dict = fcg.calculate_weights()
    weights = [weights_dict[shot] for shot in shots]
    
    # Plasma current
    plt.figure('Current')
    plt.hist(current, label='non-weighted')
    plt.hist(current, weights=weights, color='g', label='weighted')
    plt.xlabel('$I_p$ (MA)')
    plt.legend()
    
    # Toroidal B-field
    plt.figure('B-field')
    plt.hist(B_field, label='non-weighted')
    plt.hist(B_field, weights=weights, color='g', label='weighted')
    plt.xlabel('$B_t$ (T)')
    plt.legend()
    
    # NBI power
    plt.figure('NBI power')
    plt.hist(nbi_pow, label='non-weighted')
    plt.hist(nbi_pow, weights=weights, color='g', label='weighted')
    plt.xlabel('$P_{NBI}$ (MW)')
    plt.legend()

    # Electron density
    plt.figure('Electron density')
    plt.hist(e_density, label='non-weighted')
    plt.hist(e_density, weights=weights, color='g', label='weighted')
    plt.xlabel('$n_{e}$ (m$^{-3}$)')
    plt.legend()

    # Electron temperature
    plt.figure('Electron temperature')
    plt.hist(e_temp, label='non-weighted')
    plt.hist(e_temp, weights=weights, color='g', label='weighted')
    plt.xlabel('$T_{e}$ (keV)')
    plt.legend()

