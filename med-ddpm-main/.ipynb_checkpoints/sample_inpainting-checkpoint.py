import sys
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter
import torch
import sklearn
from tensorboard.backend.event_processing import event_accumulator
from diffusion_model.unet import create_model
from diffusion_model.trainer import GaussianDiffusion

sys.path.append('/workspace/MRI-inpainting-project')

from data_scripts.datasets import PathologicalMRIDataset, HealthyMRIDataset, TrainPatchesDataset
from data_scripts.visualization_utils import ImageSliceViewer3D
from dataset import TrainFCDDataset, HealthyFCDDataset, TrainFCDPatchesDataset
from dataset import reconstruct_patch, inpaint_healthy_mri
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--ckpt_path', type=str, default="")
parser.add_argument('--channel_mult', type=str, default="1,2,4,4")
parser.add_argument('--gpu_device', type=int, default=0)
parser.add_argument('--mri_dir', type=str, default="")
parser.add_argument('--mask_dir', type=str, default="")
parser.add_argument('--patch_mask_dir', type=str, default="")
parser.add_argument('--save_dir', type=str, default="")
args = parser.parse_args()

input_size=40
depth_size=40

transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),
    Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),
    Lambda(lambda t: t.transpose(3, 1)),
])

ckpt_path = args.ckpt_path
channel_mult = args.channel_mult

with torch.cuda.device(f'cuda:{args.gpu_device}'):
    model = create_model(input_size, num_channels=64, num_res_blocks=1, in_channels=3, out_channels=1, channel_mult=channel_mult).cuda()
    
    diffusion = GaussianDiffusion(
        model,
        image_size = input_size,
        depth_size = depth_size,
        timesteps = 250,   # number of steps
        loss_type = 'l1_masked', 
        with_condition=True,
    ).cuda()
    
    diffusion.load_state_dict(torch.load(ckpt_path, map_location=f'cuda:{args.gpu_device}')['ema'])
    print("Model Loaded!")

healthy_dataset = HealthyFCDDataset(args.mri_dir, args.mask_dir, args.patch_mask_dir,
                                    input_size=input_size,
                                    depth_size=depth_size,
                                    mask_transform=input_transform,
                                    mri_transform=transform,
                                    device=f'cuda:{args.gpu_device}')

split_dir = ckpt_path.split('_')[-3]
split_save_dir = os.path.join(args.save_dir, split_dir)
mri_save_dir = os.path.join(split_save_dir, 'mri')
mask_save_dir = os.path.join(split_save_dir, 'mask')

os.makedirs(split_save_dir) if not os.path.exists(split_save_dir) else None
os.makedirs(mri_save_dir) if not os.path.exists(mri_save_dir) else None
os.makedirs(mask_save_dir) if not os.path.exists(mask_save_dir) else None


for sample_id in range(len(healthy_dataset)):
    recon_mri, fcd_mask, filename = inpaint_healthy_mri(healthy_dataset, diffusion, sample_id)
    recon_mri_nib = nib.Nifti1Image(recon_mri, affine=np.eye(4))
    fcd_mask_nib = nib.Nifti1Image(fcd_mask, affine=np.eye(4))

    nib.save(recon_mri_nib, os.path.join(mri_save_dir, filename))
    nib.save(fcd_mask_nib, os.path.join(mask_save_dir, filename))






