import numpy as np
import torch
from torch.utils.data import Dataset
from torchmd_cg.nnp.schnet_dataset import SchNetDataset
from schnetpack.environment import SimpleEnvironmentProvider
from schnetpack.data.loader import _collate_aseatoms
from torch.utils.data import DataLoader
from torchmd_cg.nnp.model import make_schnet_model, load_schnet_model
from schnetpack import Properties


class External:
    def __init__(self, netfile, embeddings, device="cpu"):
        self.model = load_schnet_model(
            netfile, device=device, derivative="forces", label="energy"
        )
        self.model.to(device)
        self.device = device
        self.embeddings = embeddings.to(device)

        nreplicas = self.embeddings.shape[0]
        natoms = self.embeddings.shape[1]

        self.cell_offset = torch.zeros(
            [nreplicas, natoms, natoms - 1, 3], dtype=torch.float32
        ).to(device)

        # All vs all neighbors
        self.neighbors = torch.zeros(
            (nreplicas, natoms, natoms - 1), dtype=torch.int64
        ).to(device)
        for i in range(natoms):
            self.neighbors[:, i, :i] = torch.arange(0, i, dtype=torch.int64)
            self.neighbors[:, i, i:] = torch.arange(i + 1, natoms, dtype=torch.int64)

        self.neighbor_mask = torch.ones(
            (nreplicas, natoms, natoms - 1), dtype=torch.float32
        ).to(device)
        self.atom_mask = torch.ones((nreplicas, natoms), dtype=torch.float32).to(device)

        self.model.eval()

    def calculate(self, pos, box):
        assert pos.ndim == 3
        assert box.ndim == 3

        pos = pos.to(self.device).type(torch.float32)
        box = box.to(self.device).type(torch.float32)
        batch = {
            Properties.R: pos,
            Properties.cell: box,
            Properties.Z: self.embeddings,
            Properties.cell_offset: self.cell_offset,
            Properties.neighbors: self.neighbors,
            Properties.neighbor_mask: self.neighbor_mask,
            Properties.atom_mask: self.atom_mask,
        }
        pred = self.model(batch)
        return pred["energy"].detach(), pred["forces"].detach()


if __name__ == "__main__":
    mydevice = "cuda"
    coords = np.array(
        [
            [-6.878, -0.708, 2.896],
            [-4.189, -0.302, 0.213],
            [-1.287, 1.320, 2.084],
            [0.579, 3.407, -0.513],
            [3.531, 3.694, 1.893],
            [4.684, 0.239, 0.748],
            [2.498, -0.018, -2.375],
            [0.411, -3.025, -1.274],
            [-2.598, -4.011, 0.868],
            [-1.229, -3.774, 4.431],
        ],
        dtype=np.float32,
    ).reshape(1, -1, 3)
    coords = np.repeat(coords, 2, axis=0)
    box = np.array([56.3, 48.7, 24.2], dtype=np.float32).reshape(1, 3)
    box = np.repeat(box, 2, axis=0)
    # atom_pos = torch.tensor(coords).unsqueeze(0).to(mydevice)
    # box_t = torch.Tensor(box).unsqueeze(0).to(mydevice)
    z = np.load("../../tests/data/chignolin_aa.npy")
    z = z[:, 1].astype(np.int)
    ext = External("../../tests/data/model.ckp.30", z, mydevice)
    Epot, f = ext.calculate(coords, box)
    print(Epot)
    print(f)
