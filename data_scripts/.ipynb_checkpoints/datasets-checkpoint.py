from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np


class TrainPatchesDataset(Dataset):
    def __init__(self, dir_path, return_numpy=True, train=True, splits_filename=None, split_id=None):
        self.dir_path = dir_path
        self.filenames = sorted(os.listdir(f"{dir_path}/original"))
        self.return_numpy = return_numpy
        self.splits_filename = str(splits_filename)
        self.split_id = split_id
        self.train_or_val = 'train' if train else 'val'
        if split_id is not None:
            self.split = np.load(str(dir_path) + '/' + self.splits_filename, allow_pickle=True)[self.split_id][self.train_or_val]

    def __len__(self):
        if self.split_id is not None:
            return len(self.split)
        else:
            return len(self.filenames)

    def _getbyname(self, filename):
        orig = nib.load(f"{self.dir_path}/original/{filename}")
        label = nib.load(f"{self.dir_path}/labels/{filename}")
        opposite = nib.load(f"{self.dir_path}/opposite/{filename}")

        if os.path.isfile(f"{self.dir_path}/flair/{filename}"):
            flair = nib.load(f"{self.dir_path}/flair/{filename}")
        else:
            flair = None

        if self.return_numpy:
            return {'patch' : orig.get_fdata(), 'mask' : label.get_fdata(), 'opposite_patch' : opposite.get_fdata(), 'flair': flair.get_fdata() if flair is not None else flair, 'filename' : filename}
        else:
            return {'patch' : orig, 'mask' : label, 'opposite_patch' : opposite, 'flair': flair, 'filename' : filename}
        
    def __getitem__(self, idx):
        if self.split_id is not None:
            if idx >= len(self):
                raise StopIteration

            filename = f"sub-{self.split[idx]}.nii.gz"
        else:
            filename = self.filenames[idx]
            
        return self._getbyname(filename)

    def getbyname(self, filename):
        return self._getbyname(filename)


class PathologicalMRIDataset(Dataset):
    def __init__(self, dir_path, return_numpy=True):
        self.dir_path = dir_path
        self.filenames = sorted(os.listdir(f"{dir_path}/mri"))
        self.return_numpy = return_numpy
        self.getbyname_flag=False
        self.cur_filename = ''
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        return self._getbyname(filename)

    def getbyname(self, filename):
        return self._getbyname(filename)

    def _getbyname(self, filename):
        maskname = filename.split('-')[1].split('_')[0] + ".nii.gz"
        brain_maskname = filename.split('.')[0] + "_mask.nii.gz"
        orig = nib.load(f"{self.dir_path}/mri/{filename}")
        label = nib.load(f"{self.dir_path}/label/{maskname}")
        brain_mask = nib.load(f"{self.dir_path}/brain_mask/{brain_maskname}")
        
        if self.return_numpy:
            return {'mri' : orig.get_fdata(), 'mask' : label.get_fdata(), 'brain_mask': brain_mask.get_fdata(), 'filename' : filename}
        else:
            return {'mri' : orig, 'mask' : label, 'brain_mask': brain_mask, 'filename' : filename}
        
        
class HealthyMRIDataset(Dataset):
    def __init__(self, mri_path, mask_path=None, patch_mask_path=None, brain_mask_path=None, return_numpy=True, size=None):
        self.mri_path = mri_path
        self.mask_path = mask_path
        self.patch_mask_path = patch_mask_path
        self.brain_mask_path = brain_mask_path
        self.filenames = sorted(os.listdir(mri_path))
        self.return_numpy = return_numpy
        self.size = size

    def __len__(self):
        if self.size is None:
            return len(self.filenames)
        else:
            return self.size

    def getbyname(self, filename):
        return self._getbyname(filename)

    def _getbyname(self, filename):
        maskname = filename.split('.')[0] + "-mask.nii.gz" 
        patchmaskname = filename.split('.')[0] + "-patch-mask.nii.gz" 
        brainmaskname = filename.split('.')[0] + "_mask.nii.gz"
        
        if "_fl_" in filename: # in case of flair modality, masknames correspond to t1 filenames
            maskname = maskname.replace("_fl_", "_t1_")
            patchmaskname = patchmaskname.replace("_fl_", "_t1_")
        
        orig = nib.load(f"{self.mri_path}/{filename}")
        mask = nib.load(f"{self.mask_path}/{maskname}") if self.mask_path is not None else None
        patch_mask = nib.load(f"{self.patch_mask_path}/{patchmaskname}") if self.patch_mask_path is not None else None
        brain_mask = nib.load(f"{self.brain_mask_path}/{brainmaskname}") if self.brain_mask_path is not None else None
        
        if self.return_numpy:
            return {'mri' : orig.get_fdata(), 'mask' : mask.get_fdata() if mask is not None else mask, 
                    'patch_mask' : patch_mask.get_fdata() if patch_mask is not None else patch_mask, 
                    'brain_mask': brain_mask.get_fdata() if brain_mask is not None else brain_mask, 'filename' : filename}
        else:
            return {'mri' : orig, 'mask' : mask, 'patch_mask': patch_mask, 'brain_mask': brain_mask, 'filename' : filename}

    def __getitem__(self, idx):
        if self.size is not None and idx >= self.size:
            raise StopIteration
            
        filename = self.filenames[idx] 
        return self._getbyname(filename)

