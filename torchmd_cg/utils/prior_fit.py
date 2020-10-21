import numpy as np
import os
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from scipy.optimize import curve_fit
from moleculekit.molecule import Molecule
from moleculekit.projections.metricrmsd import MetricRmsd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


kB = 0.0019872041 # kcal/mol/K

# Input:  counts and bin limits (n+1)
# Output: counts normalized by spheric shell volumes, and bin centers (n)
def renorm(counts, bins):
    h = bins[1]-bins[0]
    R = .5*(bins[1:]+bins[:-1])  # bin centers
    vols = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)
    ncounts = counts / vols
    return np.vstack([R,ncounts])


# Bonded fit

def harmonic(x,x0,k,V0):
    return k*(x-x0)**2+V0

# Coarse Graining repulsion fit

def CG(r,eps,V0):
    sigma=1.0
    sigma_over_r = sigma/r
    V = 4*eps*(sigma_over_r**6) + V0
    return V

def get_param_bonded(mol, bond_range, Temp):
    bonds_types = {}
    for bond in mol.bonds:
        btype = tuple(mol.atomtype[bond])
        if btype in bonds_types:
            bonds_types[btype].append(bond)
        elif tuple([btype[1], btype[0]]) in bonds_types:
            bonds_types[tuple([btype[1], btype[0]])].append(bond)
        else:
            bonds_types[btype] = [bond]

    prior_bond = {}

    for bond in bonds_types.keys():
        dists = []
        for idx0, idx1 in bonds_types[bond]:
            dists.append(np.linalg.norm(mol.coords[idx0,:,:] - mol.coords[idx1,:,:], axis=0))

        dist = np.concatenate(dists, axis=0)

        yb, bins= np.histogram(dist, bins=40,  range=bond_range)
        RR, ncounts = renorm(yb, bins)

        # Drop zero counts
        RR_nz = RR[ncounts>0]
        ncounts_nz = ncounts[ncounts>0]
        dG_nz = -kB*Temp*np.log(ncounts_nz)


        # Fit may fail, better to try-catch. p0 usually not necessary if function is reasonable.
        popt, _ = curve_fit(harmonic, RR_nz, dG_nz, p0=[np.array(bond_range).mean(), 60, -1])

        # Just a hard-coded example, the full code requires more changes
        bname=f"({bond[0]}, {bond[1]})"
        prior_bond[bname]={'req': popt[0].tolist(),
                            'k0':  popt[1].tolist() }


        # popt now has the function parameters
        plt.plot(RR_nz, dG_nz, 'o')
        plt.plot(RR_nz, harmonic(RR_nz, *popt))
        plt.title(f'{bond[0]}-{bond[1]}')
        plt.show()

    return prior_bond


def get_param_nonbonded(mol, fit_range, Temp):
    atom_types = {}
    for at in set(mol.atomtype):
        atom_types[at] = np.where(mol.atomtype == at)[0]

    prior_lj = {}

    for at in atom_types.keys():
        dists = []
        for idx in atom_types[at]:
            bonded = []
            for bond in mol.bonds:
                if idx in bond:
                    bonded.append(bond[0])
                    bonded.append(bond[1])

            for idx2 in list(set(mol.bonds.flatten())):
                if idx2 not in bonded:
                    dists.append(np.linalg.norm(mol.coords[idx, :, :] - mol.coords[idx2, :, :], axis=0))

        dist = np.concatenate(dists, axis=0)
        # save only below 6A
        dist = dist[dist < fit_range[1]]

        yb, bins = np.histogram(dist, bins=30, range=fit_range)  ### Adjust the range if needed
        RR, ncounts = renorm(yb, bins)

        RR_nz = RR[ncounts > 0]
        ncounts_nz = ncounts[ncounts > 0]
        dG_nz = -kB * Temp * np.log(ncounts_nz)

        popt, _ = curve_fit(CG, RR_nz, dG_nz, p0=[2229.0, 1.])

        # Just a hard-coded example, the full code requires more changes
        bname = at
        prior_lj[bname] = {'epsilon': popt[0].tolist(),
                           'sigma': 1.0}

        plt.plot(RR_nz, dG_nz, 'o')
        plt.plot(RR_nz, CG(RR_nz, *popt))
        plt.title(f'{at}')
        plt.show()

    return prior_lj