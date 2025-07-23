from torch.utils.data import Dataset
import os
import sys
import nibabel as nib
import torch
import numpy as np

sys.path.append('../data_scripts')
from datasets import TrainPatchesDataset, PathologicalMRIDataset, HealthyMRIDataset


class TrainPatchesDatasetI2SB(Dataset):
    def __init__(self, dir_path, out_image_size=(256, 256), train=True, split_id=None, splits_filename=None, flair=False):
        self.dataset = TrainPatchesDataset(dir_path, return_numpy=True)
        self.slices_num_in_patch = self.dataset[0]['patch'].shape[2]
        self.out_image_size = out_image_size
        self.split_id = split_id
        self.train_or_val = 'train' if train else 'val'
        self.splits_filename = str(splits_filename)
        self.t1_or_flair = 'patch' if not flair else 'flair'
        if split_id is not None:
            self.split = np.load(str(dir_path) + '/' + self.splits_filename, allow_pickle=True)[self.split_id][self.train_or_val]

        self.max_pixel_val = 1457.3842 # precalculated value for normalization (max pixel value)
        self.mask_threshold = 0.5

    def __len__(self):
        if self.split_id is not None:
            return len(self.split) * self.slices_num_in_patch
        else:
            return len(self.dataset) * self.slices_num_in_patch

    def __getitem__(self, idx):
        patch_id = idx // self.slices_num_in_patch
        slice_id = idx % self.slices_num_in_patch
        if self.split_id is not None:
            if idx >= len(self):
                raise StopIteration

            patch_filename = f"sub-{self.split[patch_id]}.nii.gz"
            sample = self.dataset.getbyname(patch_filename)
        else:
            sample = self.dataset[patch_id]
            
        patch, mask = sample[self.t1_or_flair][:, :, slice_id], sample['mask'][:, :, slice_id]
        patch = torch.from_numpy(patch)
        mask = torch.from_numpy(mask)
        y = torch.tensor([0])

        patch = ((patch / self.max_pixel_val) - 0.5) * 2 # [-1, 1]
        patch = patch.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        patch = torch.nn.functional.interpolate(patch, size=self.out_image_size, mode='bilinear', align_corners=False)
        patch = patch.flatten(start_dim=0, end_dim=1)
        patch = patch.float()
        
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=self.out_image_size, mode='bilinear', align_corners=False)
        mask = (mask > self.mask_threshold).to(torch.uint8)
        mask = mask.flatten(start_dim=0, end_dim=1)

        return patch, mask, y


class HealthyPatchesDatasetI2SB(Dataset):
    def __init__(self, mri_path, mask_path, patch_mask_path, out_image_size=(256, 256), flair=False):
        self.dataset = HealthyMRIDataset(mri_path, mask_path, patch_mask_path, return_numpy=True)
        self.slices_num_in_patch = 40 # to fix
        self.out_image_size = out_image_size

        self.max_pixel_val = 1457.3842 if not flair else 398.2227  # precalculated value for normalization (max pixel value)
        self.mask_threshold = 0.5

    def __len__(self):
        return len(self.dataset) * self.slices_num_in_patch

    def __getitem__(self, idx):
        patch_id = idx // self.slices_num_in_patch
        slice_id = idx % self.slices_num_in_patch
        sample = self.dataset[patch_id]
            
        mri, mask, patch_mask = sample['mri'], sample['mask'], sample['patch_mask']

        patch = mri[patch_mask.astype(bool)].reshape(40, 40, -1)
        mask = mask[patch_mask.astype(bool)].reshape(40, 40, -1)

        #patch, mask = patch[:, :, 12:28], mask[:, :, 12:28]

        patch, mask = patch[:, :, slice_id], mask[:, :, slice_id]
        
        patch = torch.from_numpy(patch)
        mask = torch.from_numpy(mask)
        y = torch.tensor([0])

        patch = ((patch / self.max_pixel_val) - 0.5) * 2 # [-1, 1]
        patch = patch.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        patch = torch.nn.functional.interpolate(patch, size=self.out_image_size, mode='bilinear', align_corners=False)
        patch = patch.flatten(start_dim=0, end_dim=1)
        patch = patch.float()
        
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=self.out_image_size, mode='bilinear', align_corners=False)
        mask = (mask > self.mask_threshold).to(torch.uint8)
        mask = mask.flatten(start_dim=0, end_dim=1)

        return patch, mask, y

class ReconMRIDataset(Dataset):
    def __init__(self, recons_path, healthy_dataset, flair=False):
        self.recons = torch.load(recons_path)['arr']
        self.healthy_dataset = healthy_dataset
        self.slices_num_in_patch = 40 
        self.max_pixel_val = 1457.3842 if not flair else 398.2227

    def __len__(self):
        return len(self.healthy_dataset)

    def __getitem__(self, idx):
        healthy_sample = self.healthy_dataset[idx]
        
        recon = self.recons[self.slices_num_in_patch * idx: self.slices_num_in_patch * (idx+1)]
        recon = torch.nn.functional.interpolate(recon, size=(40, 40), mode='bilinear', align_corners=False)

        mri, patch_mask, mask, filename = healthy_sample['mri'], healthy_sample['patch_mask'], healthy_sample['mask'], healthy_sample['filename']
        # left = np.where(patch_mask > 0)[2][12]
        # right = np.where(patch_mask > 0)[2][28]
        
        # patch_mask[:, :, :left] = 0.0
        # patch_mask[:, :, right:] = 0.0

        recon_norm = (((recon.mean(axis=1) / 2) + 0.5) * self.max_pixel_val)

        mri_recon = mri.copy()
        mri_recon[patch_mask > 0] = np.transpose(recon_norm, (1, 2, 0)).flatten()

        return {'mri_recon': mri_recon, 'mask': mask, 'filename': filename}
