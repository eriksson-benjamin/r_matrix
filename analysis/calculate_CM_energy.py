#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 08:22:01 2023

@author: beriksso

Calculate the ion energy distributions (thermal T and NBI slowing down in 
thermal T) for all discharges and given time slices. Calculate the effective CM
energy distribution of the T+T reactions adjusted by sigma*v. Plot and save 
results to json file.
"""


import numpy as np
from nes.utils import jetplasma
from nes.utils import reactions
from nes.utils import spec
import useful_defs as udfs
udfs.set_nes_plot_style()
from nes.utils import jetnbi
import matplotlib.pyplot as plt
import sys
import scipy.constants as cst


def read_xsec(energy_axis, COFM=True):
    """
    Read T(T,2n)He4 cross section data.

    Parameters
    ----------
    energy_axis : array_like,
        The energy axis in keV.
    COFM : bool, optional,
        Whether or not to convert to the center-of-mass frame. Default is True.

    Returns
    -------
    xs : array_like,
        The cross section data in m^2.
    """
    filename = f'input_files/cross_sections/T(T,2n)He4.txt'
    
    # E is in eV, xs is in barn
    E, xs, _ = np.genfromtxt(filename, comments='#', unpack=True)
    if COFM:
        m1 = cst.physical_constants['triton mass in u'][0]
        m2 = cst.physical_constants['triton mass in u'][0]
        E *= m1 / (m1 + m2)

    # E_CM is in keV, xs is translated to m^2
    xs = np.interp(energy_axis, E*1E-3, xs*1E-28)
    return xs


def plot_temperatures(shots, t0, t1):
    """Plot ion and electron temperatures for given shots and time slices."""
    # Check ion temperatures
    for shot, t_i, t_f in zip(shots, t0, t1):
        # Get ion temperature
        Ti, time_i = udfs.get_CXRS_Ti(shot, time_range=(t_i, t_f))
        
        plt.figure(shot)
        plt.plot(time_i, Ti, label='Ti')
        print(f'{shot}')
        print('------')
        print(f'Ti = {Ti.mean():.2f} +/- {Ti.std():.2f}')
        
        # Get electron temperatures
        Te = udfs.get_Te(shot, time_range=(t_i, t_f))
        
        # Loop over Te diagnostics
        for key, val in Te.items():
            plt.plot(val[0], val[1], label=f'Te ({key})')
            print(f'Te = {val[1].mean():.2f} +/- {val[1].std():.2f}')
        print('')

        plt.title(f'{shot}', loc='left')
        plt.legend()

    udfs.multifig('temperatures', combine_pdf=True)
    
   
def _calculate_Ecm(E_th, E_nb):
    """Calculate CM energy (keV) for reaction TH and NB populations."""
    # Masses (eV)
    m1 = cst.physical_constants['triton mass energy equivalent in MeV'][0]*1E6
    m2 = cst.physical_constants['triton mass energy equivalent in MeV'][0]*1E6
    
    # Particle kinetic energies (eV)
    E1 = 1E3 * E_th
    E2 = 1E3 * E_nb
    
    # Total energies (E_k + m)
    Etot_1 = E1 + m1
    Etot_2 = E2 + m2
    
    # Invariant mass (eV)
    M = np.sqrt(m1**2 + m2**2 + 2*Etot_1*Etot_2 + 
                2 * np.sqrt(Etot_1**2 - m1**2)*np.sqrt(Etot_2**2 - m2**2))
    
    # CM energy
    Ecm = (M - (m1 + m2)) / 1000  # keV
    
    return Ecm


def calculate_Ecm(th, nb):
    """
    Calculate CM energy (keV) and relative velocities (m/s) for reactions 
    between thermal and beam populations.
    """
    # Get the sampled four-vectors
    P_tot = th.P + nb.P
    
    # Invariant mass
    M = np.sqrt(P_tot[0]**2 - np.sum(P_tot[1:]**2, axis=0))
    
    # CM energies
    Ecm = M - (th.m + nb.m)
    
    # Relative velocities
    v_rel = th.v - nb.v
    V = np.sqrt(np.sum(v_rel**2, axis=0))
    
    return Ecm, V


def fuel_ion_dists(shot, t0, t1):
    """Return reactant objects for thermal and beam T ion distributions."""
    # Compute T NBI slowing down in thermal T plasma
    ne, Te, B, R0 = jetplasma.get_params(shot, t0, t1)
    E, D = jetnbi.get_dist(shot, t0, t1, ion='T')
    
    # Calculate tritium NBI on T plasma BT components
    tt_reaction = reactions.TT2NHe4Reaction()
    tt_scalc = spec.SpectrumCalculator(tt_reaction)
    tt_scalc.u1 = [0, 0, 1]
    
    # Set reactant velocity distributions
    tt_scalc.reactant_b.sample_E_dist(E, D, pitch_range=[0.5, 0.7])  # b = triton
    tt_scalc.reactant_a.sample_maxwellian_dist(Te)  # a = triton
    
    # Return reactant objects
    th = tt_scalc.reactant_a
    nb = tt_scalc.reactant_b
    
    return th, nb


def plot_Elab(bin_edges, E_th, E_nb):
    """Plot the lab T energy distributions."""
    # Make histograms
    bin_centres = bin_edges[1:] - np.diff(bin_edges)[0] / 2
    h_th, _ = np.histogram(E_th, bins=bin_edges)
    h_nb, _ = np.histogram(E_nb, bins=bin_edges)
        
    # Plot
    plt.figure('Fuel ion distributions')
    plt.plot(bin_centres, h_th, 'k--', label='Thermal T')
    plt.plot(bin_centres, h_nb * h_th.max() / h_nb.max(), 'C0-', label='NB T')
    plt.xlabel('Fuel ion $E_{lab}$ (keV)')
    plt.ylabel('Fuel density (a.u.)')
    plt.legend()
    
    return bin_centres, h_th, h_nb


def plot_ECM(bin_centres, Ecm_h, Ecm_xs, xs):
    """Plot the CM energy distributions together with T+T cross section."""
    plt.figure('CM energy distributions')
    
    # Plot CM E dist. and cross section adjusted distribution
    plt.plot(bin_centres, Ecm_h, 'C0-.', label='T+T')
    plt.plot(bin_centres, Ecm_xs * Ecm_h.max() / Ecm_xs.max(), 'C1-',
             label='T+T adjusted')
    plt.xlabel('Fuel ion $E_{CM}$ (keV)')
    plt.ylabel('Rel. intensity (a.u.)')
    ax1 = plt.gca()
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 3E10)
    ax1.set_xlim(0, 150)

    # Calculate 1 sigma intervals
    cs = np.cumsum(Ecm_xs) / Ecm_xs.sum()
    mean = bin_centres[np.argmin(np.abs(cs - 0.5))]
    std_l = bin_centres[np.argmin(np.abs(cs - 0.15865))]
    std_u = bin_centres[np.argmin(np.abs(cs - (1 - 0.15865)))]
    
    print(f'mean = {mean:.2f} +({std_u - mean:.2f})/-({mean - std_l:.2f})')
    print(f'std_l = {std_l:.2f}')
    print(f'std_h = {std_u:.2f}')
    
    # Plot intervals
    plt.axvline(mean, color='C1', linestyle='--')
    plt.axvline(std_l, color='C1', linestyle='--')
    plt.axvline(std_u, color='C1', linestyle='--')

    # Plot cross section
    ax2 = ax1.twinx()
    ax2.plot(bin_centres, xs * 1E28, 'r--', label='T+T cross section')
    ax2.set_ylabel('Cross section (barn)', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 150)
    ax2.set_ylim(0, 0.06)
    
    return mean, std_l, std_u


def main(shot, t0, t1, bin_edges):
    # Get fuel ion energy distributions
    th, nb = fuel_ion_dists(shot, t0, t1)

    # Plot distributions
    bin_centres, h_th, h_nb = plot_Elab(bin_edges, th.E, nb.E)
    
    # Calculate CM energy of the samples
    Ecm, V = calculate_Ecm(th, nb)
    
    # Read cross section for TT reaction
    xs = read_xsec(bin_centres, True)
    
    # Calculate the cross section (sigma*v) adjusted CM energy distribution
    hcm, _ = np.histogram(Ecm, bins=bin_edges, weights=V)
    hcm_xs = hcm * xs
    
    plt.figure()
    weights1 = read_xsec(Ecm, True) * V
    hhcm, _ = np.histogram(Ecm, bins=bin_edges, weights=weights1)
    plt.plot(bin_centres, hcm_xs/hcm_xs.max(), linewidth=4, label='hcm')
    plt.plot(bin_centres, hhcm/hhcm.max(), linestyle='-', label='hhcm')
    plt.legend()
    
    plt.figure()
    we, _ = np.histogram(weights1)
    plt.plot()
    
    # Plot CM distributions
    mean, std_l, std_u = plot_ECM(bin_centres, hcm, hcm_xs, xs)

    return mean, std_l, std_u, h_th, h_nb, hcm, hcm_xs


if __name__ == '__main__':  
    # Load shot numbers
    name = 'nbi'
    discharges = np.loadtxt(f'../data/{name}/shots_{name}.txt')
    shots = np.array(discharges[:, 0], dtype='int')
    t0s, t1s = discharges[:, 1], discharges[:, 2]
    
    # Energy bins
    bin_edges = np.arange(0, 150, 0.5)
    bin_centres = udfs.get_bin_centres(bin_edges)
    
    # Dictionary to store results in
    results = {'shots': [], 'E lab dist': [], 'E CM dist': [], 'mean': [], 
               'failed':[], 'bin_centres': bin_centres.tolist()}
    counter = 1
    for shot, t0, t1 in zip(shots, t0s, t1s):
        try:
            mean, std_l, std_u, h_th, h_nb, hcm, hcm_xs = main(shot, t0, t1, 
                                                               bin_edges)
            results['shots'].append(int(shot))
            results['mean'].append([mean, std_l, std_u])
            results['E lab dist'].append([h_th.tolist(), h_nb.tolist()])
            results['E CM dist'].append([hcm.tolist(), hcm_xs.tolist()])
            
        except:
            print(f'{shot} failed.')
            results['failed'].append(int(shot))        
        
        # Write results to file
        udfs.json_write_dictionary('energy_distributions.json', results, 
                                   check=False)
        print(f'{counter}/{len(shots)} done.')
        counter += 1
