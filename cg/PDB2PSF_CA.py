import numpy as np
from moleculekit.molecule import Molecule
from cg.mappings import CA_MAP

def PDB2PSF_CA(pdb_name_in, psf_name_out, bonds=True, angles = True):
    mol = Molecule(pdb_name_in)
    mol.filter('name CA')

    n = mol.numAtoms

    atom_types = []
    for i in range(n):
        atom_types.append(CA_MAP[(mol.resname[i], mol.name[i])])

    if bonds:
        bonds = np.concatenate((np.arange(n-1).reshape([n-1, 1]), (np.arange(1, n).reshape([n-1, 1]))), axis=1)
    else:
        bonds = np.empty([0,2], dtype=np.int32)
    
    if angles:
        angles = np.concatenate((np.arange(n-2).reshape([n-2, 1]), (np.arange(1, n-1).reshape([n-2, 1])), (np.arange(2, n).reshape([n-2, 1]))), axis=1)
    else:
        angles = np.empty([0,3], dtype=np.int32)
    
    mol.atomtype = np.array(atom_types)
    mol.bonds = bonds
    mol.angles = angles
    mol.write(psf_name_out)
