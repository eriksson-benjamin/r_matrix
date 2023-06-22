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
