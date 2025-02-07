#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import sklearn
import scipy
from scipy.ndimage import gaussian_filter
import torch
import re
import os

import sys
sys.path.append('/workspace/MRI-inpainting-project')
from data_scripts.datasets import PathologicalMRIDataset, HealthyMRIDataset, TrainPatchesDataset
import scipy

def reconstruct_patch(orig_patch, gen_patch, fcd_mask):
    gen_patch = torch.from_numpy(gen_patch).transpose(0, 2).squeeze()
    gen_patch = ((gen_patch + 1) / 2).numpy()  #[-1, 1] -> [0, 1]

    orig_patch_clipped = np.clip(orig_patch, np.quantile(orig_patch, 0.001), np.quantile(orig_patch, 0.999))
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(orig_patch_clipped.reshape(-1, orig_patch_clipped.shape[-1]))
    gen_patch = scaler.inverse_transform(gen_patch.reshape(-1, gen_patch.shape[-1])).reshape(gen_patch.shape) # MinMax scaler inversed
    
    recon_patch = orig_patch.copy()
    recon_patch[fcd_mask > 0.5] = gen_patch[fcd_mask > 0.5].flatten()
    recon_patch_smoothed = gaussian_filter(recon_patch, sigma=0.5)
    recon_patch[fcd_mask > 0.5] = recon_patch_smoothed[fcd_mask > 0.5].flatten()

    return recon_patch

def inpaint_healthy_mri(medddpm_healthy_dataset, model, sample_id):
    sample = medddpm_healthy_dataset[sample_id][0]
    filename = medddpm_healthy_dataset[sample_id][1]
    orig_sample = super(type(medddpm_healthy_dataset), medddpm_healthy_dataset).getbyname(filename)
    orig_patch = orig_sample['mri'][orig_sample['patch_mask'] > 0].reshape(40, 40, 40)
    orig_fcd_mask = orig_sample['mask'][orig_sample['patch_mask'] > 0].reshape(40, 40, 40)
    
    gen_patch = model.sample(batch_size=1, condition_tensors=sample)
    gen_patch = gen_patch.cpu().numpy().squeeze()
    recon_patch = reconstruct_patch(orig_patch, gen_patch, orig_fcd_mask)

    recon_mri = orig_sample['mri'].copy()
    recon_mri[orig_sample['patch_mask'] > 0] = recon_patch.flatten()

    return recon_mri, orig_sample['mask'], filename

class NiftiImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, depth_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder, '*.nii.gz'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot_samples(self, n_slice=15, n_row=4):
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                sample = sample[0]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice])
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d= img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 3,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            pairs.append((input_file, target_file))
        return pairs

    def label2masks(self, masked_img):
        result_img = np.zeros(masked_img.shape + ( self.input_channel - 1,))
        result_img[masked_img==LabelEnum.BRAINAREA.value, 0] = 1
        result_img[masked_img==LabelEnum.TUMORAREA.value, 1] = 1
        return result_img

    def read_image(self, file_path, pass_scaler=False):
        img = nib.load(file_path).get_fdata()
        if not pass_scaler:
            img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)

        target_img = self.read_image(target_file)
        target_img = self.resize_img(target_img)

        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input': input_img, 'target': target_img}


class TrainFCDDataset(PathologicalMRIDataset):
    def __init__(self, input_dir, input_size, depth_size, mask_transform, target_transform, inpainting=False):
        super().__init__(input_dir, return_numpy=True)
        self.input_size = input_size
        self.depth_size = depth_size
        self.scaler = MinMaxScaler()
        self.mask_transform = mask_transform
        self.target_transform = target_transform
        self.inpainting = inpainting

    def __len__(self):
        return super().__len__()

    def cut_resize(self, x, y, z, image):
        image_cut = image[x.min():x.max(), y.min():y.max(), z.min():z.max()]
        resize_out_shape = (self.input_size, self.input_size, self.input_size)
        image_resized = torch.nn.functional.interpolate(torch.from_numpy(image_cut).unsqueeze(0).unsqueeze(0), 
                                                        size=resize_out_shape, mode='trilinear', align_corners=False).numpy().squeeze()

        return image_resized
        
    def sample_conditions(self, batch_size):                      # for batch_size == 1
        index = 22 #np.random.randint(0, len(self))
        sample = super().__getitem__(index) 
        mri, fcd_mask, brain_mask, filename = sample['mri'], sample['mask'], sample['brain_mask'], sample['filename']

        x, y, z = np.where(brain_mask != 0)
        mri_resized = self.cut_resize(x, y, z, mri)
        fcd_mask_resized = self.cut_resize(x, y, z, fcd_mask)
        brain_mask_resized = self.cut_resize(x, y, x, brain_mask)

        mri_resized = np.clip(mri_resized, np.quantile(mri_resized, 0.001), np.quantile(mri_resized, 0.999))
        mri_resized = self.scaler.fit_transform(mri_resized.reshape(-1, mri_resized.shape[-1])).reshape(mri_resized.shape)
        fcd_mask_resized = (fcd_mask_resized > 0.5).astype(np.float32)
        brain_mask_resized = (brain_mask_resized > 0.5).astype(np.float32)

        mri_final = self.target_transform(mri_resized)
        mask_final = self.mask_transform(np.stack((brain_mask_resized, fcd_mask_resized), axis=-1))

        if self.inpainting:
            mask_final[0] = mri_final[0] 
            mask_final[0][mask_final[1] > 0] = -1 #masking pathology

        return mask_final.unsqueeze(0).cuda()

    def __getitem__(self, idx):
        sample = super().__getitem__(idx) 
        mri, fcd_mask, brain_mask, filename = sample['mri'], sample['mask'], sample['brain_mask'], sample['filename']

        assert mri.shape == fcd_mask.shape == (197, 233, 189) 

        x, y, z = np.where(mri != 0)
        mri_resized = self.cut_resize(x, y, z, mri)
        fcd_mask_resized = self.cut_resize(x, y, z, fcd_mask)
        brain_mask_resized = self.cut_resize(x, y, x, brain_mask)
        
        fcd_mask_resized = (fcd_mask_resized > 0.5).astype(np.float32)
        brain_mask_resized = (brain_mask_resized > 0.5).astype(np.float32)
        mri_resized = np.clip(mri_resized, np.quantile(mri_resized, 0.001), np.quantile(mri_resized, 0.999))
        mri_resized = self.scaler.fit_transform(mri_resized.reshape(-1, mri_resized.shape[-1])).reshape(mri_resized.shape)

        mri_final = self.target_transform(mri_resized)
        mask_final = self.mask_transform(np.stack((brain_mask_resized, fcd_mask_resized), axis=-1))

        if self.inpainting:
            mask_final[0] = mri_final[0] 
            mask_final[0][mask_final[1] > 0] = -1 #masking pathology

        return {'input': mask_final, 'target': mri_final}


class HealthyFCDDataset(HealthyMRIDataset):
    def __init__(self, mri_path, mask_path=None, patch_mask_path=None, brain_mask_path=None, return_numpy=True, size=None,
                input_size=128, depth_size=128, mask_transform=None, mri_transform=None, inpainting=True, device='cuda:1'):
        super().__init__(mri_path, mask_path, patch_mask_path, brain_mask_path, return_numpy, size)
        self.input_size = input_size
        self.depth_size = depth_size
        self.scaler = MinMaxScaler()
        self.mask_transform = mask_transform
        self.mri_transform = mri_transform
        self.inpainting = inpainting
        self.device = device

    def __len__(self):
        return super().__len__()

    def cut_resize(self, x, y, z, image):
        image_cut = image[x.min():x.max(), y.min():y.max(), z.min():z.max()]
        resize_out_shape = (self.input_size, self.input_size, self.input_size)
        image_resized = torch.nn.functional.interpolate(torch.from_numpy(image_cut).unsqueeze(0).unsqueeze(0), 
                                                        size=resize_out_shape, mode='trilinear', align_corners=False).numpy().squeeze()

        return image_resized, image_cut.shape

    def __getitem__(self, idx):
        sample = super().__getitem__(idx) 
        mri, fcd_mask, brain_mask, patch_mask = sample['mri'], sample['mask'], sample['brain_mask'], sample['patch_mask']
        filename = sample['filename']

        if patch_mask is None:
            x, y, z = np.where(brain_mask != 0)
            mri_resized, cut_shape = self.cut_resize(x, y, z, mri)
            fcd_mask_resized, _ = self.cut_resize(x, y, z, fcd_mask)
            brain_mask_resized, _ = self.cut_resize(x, y, x, brain_mask)
        else:
            mri_resized = mri[patch_mask > 0].reshape(self.input_size, self.input_size, self.depth_size)
            fcd_mask_resized = fcd_mask[patch_mask > 0].reshape(self.input_size, self.input_size, self.depth_size)
            brain_mask_resized = np.zeros_like(fcd_mask_resized)
            cut_shape = None

        mri_resized = np.clip(mri_resized, np.quantile(mri_resized, 0.001), np.quantile(mri_resized, 0.999))
        min_val, max_val = mri_resized.min(), mri_resized.max()
        mri_resized = self.scaler.fit_transform(mri_resized.reshape(-1, mri_resized.shape[-1])).reshape(mri_resized.shape)
        fcd_mask_resized = (fcd_mask_resized > 0.5).astype(np.float32)
        brain_mask_resized = (brain_mask_resized > 0.5).astype(np.float32)

        mri_final = self.mri_transform(mri_resized)
        mask_final = self.mask_transform(np.stack((brain_mask_resized, fcd_mask_resized), axis=-1))

        if self.inpainting:
            mask_final[0] = mri_final[0] 
            mask_final[0][mask_final[1] > 0] = -1 #masking pathology

        return mask_final.unsqueeze(0).cuda(self.device), filename, cut_shape, min_val, max_val    #mask_final.cuda()


class TrainFCDPatchesDataset(TrainPatchesDataset):
    def __init__(self, input_dir, input_size, depth_size, mask_transform, target_transform, splits_filename=None, split_id=None, train=True):
        super().__init__(input_dir, splits_filename=splits_filename, split_id=split_id, train=train)
        self.input_size = input_size
        self.depth_size = depth_size
        self.scaler = MinMaxScaler()
        self.mask_transform = mask_transform
        self.target_transform = target_transform

    def __len__(self):
        return super().__len__()

    def sample_conditions(self, batch_size):
        index = 0 #np.random.randint(0, len(self))
        sample = super().__getitem__(index) 
        patch, fcd_mask = sample['patch'], sample['mask']

        patch = np.clip(patch, np.quantile(patch, 0.001), np.quantile(patch, 0.999))
        min_val, max_val = patch.min(), patch.max()
        patch = self.scaler.fit_transform(patch.reshape(-1, patch.shape[-1])).reshape(patch.shape)

        fcd_mask = (fcd_mask > 0.5).astype(np.float32)

        patch_final = self.target_transform(patch)
        fcd_mask_final = self.mask_transform(np.stack((fcd_mask, fcd_mask), axis=-1))
        fcd_mask_final[0] = patch_final[0] 
        fcd_mask_final[0][fcd_mask_final[1] > 0] = -1 #masking pathology

        return fcd_mask_final.unsqueeze(0).cuda()
        
    def __getitem__(self, idx):
        sample = super().__getitem__(idx) 
        patch, fcd_mask = sample['patch'], sample['mask']

        patch = np.clip(patch, np.quantile(patch, 0.001), np.quantile(patch, 0.999))
        min_val, max_val = patch.min(), patch.max()
        patch = self.scaler.fit_transform(patch.reshape(-1, patch.shape[-1])).reshape(patch.shape)

        fcd_mask = (fcd_mask > 0.5).astype(np.float32)

        patch_final = self.target_transform(patch)
        fcd_mask_final = self.mask_transform(np.stack((fcd_mask, fcd_mask), axis=-1))
        fcd_mask_final[0] = patch_final[0] 
        fcd_mask_final[0][fcd_mask_final[1] > 0] = -1 #masking pathology

        return {'input': fcd_mask_final, 'target': patch_final}

