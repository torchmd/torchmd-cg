import torch
from moleculekit.molecule import Molecule
from mappings import CACB_MAP

def PDB2PSF_CACB(pdb_name_in, psf_name_out, bonds=True, angles = True):
    mol = Molecule(pdb_name_in)

    n = mol.numAtoms

    atom_types = []
    for i in range(n):
        atom_types.append(CACB_MAP[(mol.resname[i], mol.name[i])])
        
    CA_idx = []
    CB_idx = []
    for i, name in enumerate(mol.name):
        if name[:2] == 'CA':
            CA_idx.append(i)
        else:
            CB_idx.append(i)
    

    if bonds:
        CA_bonds = np.concatenate((np.array(CA_idx)[:-1].reshape([len(CA_idx)-1,1]), 
                                   np.array(CA_idx)[1:].reshape([len(CA_idx)-1,1])), 
                                  axis = 1)
        CB_bonds = np.concatenate((np.array(CB_idx).reshape(len(CB_idx),1)-1, 
                                   np.array(CB_idx).reshape(len(CB_idx),1)), axis=1)
        bonds = np.concatenate((CA_bonds, CB_bonds))
    else:
        bonds = np.empty([0,2], dtype=np.int32)
    
    if angles:
        CA_angles = np.concatenate((np.array(CA_idx)[:-2].reshape([len(CA_idx)-2,1]), 
                                np.array(CA_idx)[1:-1].reshape([len(CA_idx)-2,1]), 
                                np.array(CA_idx)[2:].reshape([len(CA_idx)-2,1])), axis = 1)
        
        CB_angles = []
        cbn = 0
        for i in CA_idx:
            if mol.resname[i] != 'GLY':
                if i != 0:
                    CB_angles.append([CA_idx[CA_idx.index(i)-1], i, CB_idx[cbn]])
                if i != CA_idx[-1]:
                    CB_angles.append([CA_idx[CA_idx.index(i)+1], i, CB_idx[cbn]])
                cbn+=1
        CB_angles = np.array(CB_angles)
        angles = np.concatenate((CA_angles, CB_angles))
        
    else:
        angles = np.empty([0,3], dtype=np.int32)
    
    mol.atomtype = np.array(atom_types)
    mol.bonds = bonds
    mol.angles = angles
    mol.write(psf_name_out)