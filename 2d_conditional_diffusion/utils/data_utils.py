import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
import nibabel as nib



#--------------------------------------------------------------------------------------------------------------
# Datasets


class TrainPatchDataset(Dataset):
    def __init__(self, path: str, exclude=True):
        self.path = path
        self.patches_path = path + '/original'
        self.masks_path = path + '/labels'
        self.symms_path = path + '/opposite'
        
        patients_to_exclude = []
        if exclude:
            patients_to_exclude = ['n3', 'n11', 'n13', 'G061', 'G081', 'G258', 'n23', 'n37', 'n64', 'n70']
            patients_to_exclude = ['sub-' + patient + '.nii.gz' for patient in patients_to_exclude]

        patients = [patient for patient in os.listdir(self.patches_path) 
                    if 'sub' in patient and patient not in patients_to_exclude]

        self.patches = [self.patches_path + '/' + patient for patient in patients]
        self.masks = [self.masks_path + '/' + patient for patient in patients]
        self.symm_patches = [self.symms_path + '/' + patient for patient in patients]

        assert len(self.patches) == len(self.masks) == len(self.symm_patches)

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = nib.load(self.patches[idx]).get_fdata()
        mask = nib.load(self.masks[idx]).get_fdata()
        symm_patch = nib.load(self.symm_patches[idx]).get_fdata() 

        return patch, mask, symm_patch
    
    def save_as_jpeg(self, slice, path):
        plt.imsave(path, slice, cmap='gray')


class HealthyBrainsDataset(Dataset):
    def __init__(self, brains_path: str, masks_path: str, patch_masks_path: str, symm_patch_masks_path=None):
        self.brains = np.load(brains_path)
        self.masks = torch.load(masks_path)
        self.masks = np.stack([np.stack(self.masks[i:i+40], axis=2) for i in range (0, len(self.masks), 40)])
        self.patch_masks = torch.load(patch_masks_path)
        self.symm_patch_masks = torch.load(symm_patch_masks_path) if symm_patch_masks_path is not None else None
        #assert self.brains.shape == self.masks.shape == self.patch_masks.shape
        assert len(self.brains.shape) == 4

    def __len__(self):
        return len(self.brains)
    
    def __getitem__(self, idx):
        if self.symm_patch_masks is not None:
            return self.brains[idx], self.masks[idx], self.patch_masks[idx], self.symm_patch_masks[idx]
        else:
            return self.brains[idx], self.masks[idx], self.patch_masks[idx], None
    
    def get_patch(self, idx):
        brain, mask, patch_mask, symm_patch_mask = self[idx]
        patch = brain[patch_mask]
        mask_in_patch = mask[patch_mask]
        symm_patch = brain[symm_patch_mask] if self.symm_patch_masks is not None else None

        return patch, mask_in_patch, symm_patch
    
#-----------------------------------------------------------------------------------------------------------------

def incorporate_gen_patches(healthy_brains_dataset: Dataset, gen_patches: np.array):
    '''
    Incorporates generated 3d patches into the healthy brains:
    
    Output:
    np.array of brains with inserted patches, (n_brains, x, y, z)
    '''
    
    gen_brains = []

    for ((brain, mask, patch_mask, _), gen_patch) in zip(healthy_brains_dataset, gen_patches):
        gen_brain = brain.copy()
        intensity_diff = brain[patch_mask].max()
        gen_brain[patch_mask] = (gen_patch * intensity_diff).flatten()
        patch = gen_brain[patch_mask].reshape(40, 40, 40)
        patch[(1 - mask).astype(bool)] = brain[patch_mask].reshape(40, 40, 40)[(1 - mask).astype(bool)].flatten()
        gen_brain[patch_mask] = patch.flatten()
        gen_brains.append(gen_brain)
    
    return np.stack(gen_brains)
