import torch
import glob
import numpy as np
from torch.utils.data import Dataset

class NPDataset(Dataset):
    def __init__(self,coordfile,forcefile,atomtypefile):
        coo = np.load(coordfile)
        f = np.load(forcefile)
        at = np.load(atomtypefile)
        self.coo = torch.from_numpy(coo)
        self.f = torch.from_numpy(f)
        z = at[:,1].astype(np.int)
        self.z = torch.from_numpy(z) #get only the numbers
        assert self.coo.shape == self.f.shape
        assert self.coo.shape[1] == self.z.shape[0]

    def __len__(self):
        return self.coo.shape[0]
        
    def __getitem__(self,idx):
        return {'coordinates':self.coo[idx],'forces':self.f[idx],'atomic_numbers':self.z}


class NpysDataset(Dataset):
    def __init__(self,coordfiles,forcefiles,embedfiles):
        coordfiles=sorted(glob.glob(coordfiles))
        forcefiles=sorted(glob.glob(forcefiles))
        embedfiles=sorted(glob.glob(embedfiles))
        self.coords = []
        self.forces = []
        self.embeddings = []
        self.index= []
        assert len(coordfiles)==len(forcefiles)==len(embedfiles)
        print("Coordinates ", coordfiles)
        print("Forces ", forcefiles)
        print("Embeddings ",embedfiles)	
        nfiles = len(coordfiles)
        for i in range(nfiles):
                cdata = torch.tensor( np.load(coordfiles[i]) )
                self.coords.append(cdata)
                fdata = torch.tensor( np.load(forcefiles[i]) )
                self.forces.append(fdata)
                edata = torch.tensor( np.load(embedfiles[i]).astype(np.int) )
                self.embeddings.append(edata)
                size = cdata.shape[0]
                self.index.extend(list(zip(size*[i],range(size))))
                assert cdata.shape == fdata.shape, "{} {}".format(cdata.shape, fdata.shape)
                assert cdata.shape[1] == edata.shape[0]
        print("Combined dataset size {}".format(len(self.index)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self,idx):
        fileid,index = self.index[idx]
        
        return  {'coordinates':self.coords[fileid][index],
                 'forces':self.forces[fileid][index],
                 'atomic_numbers':self.embeddings[fileid]}




class NpysDataset2(Dataset):
    def __init__(self,coordglob,forceglob,embedglob):
        self.coordfiles=sorted(glob.glob(coordglob))
        self.forcefiles=sorted(glob.glob(forceglob))
        self.embedfiles=sorted(glob.glob(embedglob))
        self.index= []
        assert len(self.coordfiles)==len(self.forcefiles)==len(self.embedfiles)
        print("Coordinates files: ", len(self.coordfiles))
        print("Forces files: ", len(self.forcefiles))
        print("Embeddings files: ", len(self.embedfiles))	
        #make index
        nfiles = len(self.coordfiles)
        for i in range(nfiles):
                cdata = np.load(self.coordfiles[i]) 
                fdata = np.load(self.forcefiles[i]) 
                edata = np.load(self.embedfiles[i]).astype(np.int) 
                size = cdata.shape[0]
                self.index.extend(list(zip(size*[i],range(size))))
                #consistency check
                assert cdata.shape == fdata.shape, "{} {}".format(cdata.shape, fdata.shape)
                assert cdata.shape[1] == edata.shape[0]
        print("Combined dataset size {}".format(len(self.index)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self,idx):
        fileid,index = self.index[idx]

        cdata = np.load(self.coordfiles[fileid], mmap_mode='r') #I only need one element
        fdata = np.load(self.forcefiles[fileid], mmap_mode='r') 
        edata = np.load(self.embedfiles[fileid]).astype(np.int) 
        
        return  {'coordinates':np.array(cdata[index]),
                 'forces':np.array(fdata[index]),
                 'atomic_numbers':edata}
