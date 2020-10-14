import torch
import numpy as np
from tqdm import tqdm
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.systems import System
from moleculekit.molecule import Molecule
from torchmd.parameters import Parameters


def make_deltaforces(
    coords_npz,
    forces_npz,
    delta_forces_npz,
    forcefield,
    psf,
    exclusions=("bonds"),
    device="cpu",
    forceterms=["Bonds", "Angles", "RepulsionCG"],
):
    device = torch.device(device)
    precision = torch.double
    replicas = 1

    mol = Molecule(psf)
    atom_types = mol.atomtype
    natoms = mol.numAtoms

    coords = np.load(coords_npz)

    ### test if any bonds are longer than 10A
    print("Check for broken coords.")

    broken_frames = []
    for bond in tqdm(mol.bonds):
        crds = coords[:, bond, :]
        crds_dif = crds[:, 0, :] - crds[:, 1, :]
        dists = np.linalg.norm(crds_dif, axis=1)
        broken = dists > 10.0
        broken_frames.append(broken)

    broken_frames = np.stack(broken_frames)
    broken_frames = broken_frames.any(axis=0)

    if broken_frames.any():
        print("Removing broken coords with distances larger than 10A.")

        coords_good = coords[~broken_frames, :, :]  # remove broken frames
        np.save(coords_npz.replace(".", "_fix."), coords_good)

        broken_coords = coords[broken_frames, :, :]
        np.save(coords_npz.replace(".", "_broken."), broken_coords)

        coords = coords_good

    else:
        print("No broken frames")

    all_forces = np.load(forces_npz)[~broken_frames, :, :]
    coords = torch.tensor(coords, dtype=precision).to(device)

    atom_vel = torch.zeros(replicas, natoms, 3)
    atom_forces = torch.zeros(natoms, 3, replicas)
    atom_pos = torch.zeros(natoms, 3, replicas)
    box_full = torch.zeros(3, replicas)

    ff = ForceField.create(mol, forcefield)
    parameters = Parameters(ff, mol, forceterms, precision=precision, device=device)

    system = System(natoms, replicas, precision, device)
    system.set_positions(atom_pos)
    system.set_box(box_full)
    system.set_velocities(atom_vel)

    forces = Forces(parameters, terms=forceterms, exclusions=exclusions)

    print("Producing delta forces")
    prior_forces = []
    for co in tqdm(coords):
        Epot = forces.compute(co.reshape([1, natoms, 3]), system.box, system.forces)
        fr = (
            system.forces.detach().cpu().numpy().astype(np.float32).reshape([natoms, 3])
        )
        prior_forces.append(fr)

    prior_forces = np.array(prior_forces)
    delta_forces = all_forces - prior_forces

    np.save(delta_forces_npz, delta_forces)
