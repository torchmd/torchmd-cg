import numpy as np
import torch
import os
from torch.utils.data import Dataset
from schnetpack import Properties
from torch.utils.data.sampler import RandomSampler

class SchNetDataset(Dataset):
    def __init__(self, dataset, environment_provider,label=['energy']):
        self.dataset = dataset
        self.environment_provider = environment_provider
        label = list(label) if isinstance(label,str) else label
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        atomic_numbers = data['atomic_numbers']
        positions = data['coordinates']
        
        properties = {}
        properties["_idx"] = torch.LongTensor([index])  
        properties[Properties.Z] = torch.LongTensor(atomic_numbers)
        properties[Properties.R] = torch.FloatTensor(positions)
        cell = torch.zeros((3,3),dtype=torch.float32)
        if 'box' in data:
            cell[torch.eye(3).bool()] = torch.FloatTensor(data['box'])
        properties[Properties.cell] = cell 

        for l in self.label:
            properties[l] = torch.FloatTensor( data[l] )

        at = FakeASE(atomic_numbers, positions, cell.numpy(), pbc=False)
        nbh_idx, offsets = self.environment_provider.get_environment(at)
        properties[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))
        properties[Properties.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))
        return properties

    # def get_label(self, label_name):
    #     labels = []
    #     natoms = []
    #     for i in range(len(self.dataset)):
    #         dset = self.dataset[i]
    #         label = dset[label_name]
    #         n_atoms = len(dset['atomic_numbers'])
    #         labels.append(label)
    #         natoms.append(n_atoms)
    #     return np.array(labels), np.array(natoms)

    # def calc_stats(self, label_name, per_atom=True):  #this is wrong, it should consider atomrefs!!!!
    #     labels, natoms = self.get_label(label_name)
    #     if per_atom:
    #         return np.mean(labels / natoms, keepdims=True, dtype='float32'), \
    #                np.std(labels / natoms, keepdims=True, dtype='float32')
    #     else:
    #         return np.mean(labels, keepdims=True, dtype='float32'), \
    #                np.std(labels, keepdims=True, dtype='float32')


class FakeASE:
    def __init__(self, numbers, positions, cell, pbc):
        self.numbers = numbers
        self.positions = positions
        self.cell = cell
        self.pbc = np.array([pbc, pbc, pbc])

    def get_number_of_atoms(self):
        return len(self.numbers)

    def get_cell(self, complete):
        return self.cell


from collections import OrderedDict 

class CachedDataset(Dataset):#TODO: UNTESTED
    def __init__(self, dataset, cache_size=200000):
        self.dataset = dataset
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __getitem__(self, idx):
        if self.cache.get(str(idx)) is not None:
            return self.cache.get(str(idx))
        else:
            if len(self.cache.keys())>=self.cache_size: 
                self.cache.popitem(last=False) #remove first  
            self.cache[str(idx)]=self.dataset[idx]
            return self.cache[str(idx)]

    def __len__(self):
        return len(self.dataset)

    def refresh(self):
        self.cache = OrderedDict()


