import os
from collections import defaultdict
import numpy as np
from monai.data import list_data_collate
import monai
from monai.data import DataLoader, Dataset 
from monai.transforms import (
    LoadImage, Spacingd, RandZoomd,
    RandFlipd, Resized, RandAffined,
    LoadImaged, EnsureChannelFirstd,
    Resized, EnsureTyped, Compose, ScaleIntensityd, 
    RandGaussianNoised, RandRotated, ToTensord
)
import torch
from torchio.transforms.preprocessing.intensity import histogram_standardization
from torchio.transforms.preprocessing.intensity import z_normalization
from pathlib import Path

from utils import  normalize_



def assign_feature_maps(sub, feature, dir, fold=None):
    '''
    Mapping from `sub` and `feature` to the corresponding path
    of feature that belongs ot this subject 
    the list of possible paths may be returned instead 
    of single path
    '''
    if feature == 'image':
        if ('00' in sub)|('C' in sub):
            if fold!=None:
                feature_map = os.path.join(dir, f'split{fold}/t1/sub-{sub}_t1_brain-final.nii.gz')
        else:
            feature_map = os.path.join(dir, f'sub-{sub}', f'sub-{sub}_t1_brain-final.nii.gz')
        
    return feature_map


def create_datafile(sub_list, feat_params, data_path, label_path, augm_data_path, mask=False, fold=None):
    
    '''
    for each subject from `sub_list` 
    collects corresponding features from `feat_params`
    and segmentation mask
    
    mask : Bool,
        Include mask key to output file
    '''

    files = []
    missing_files = defaultdict(list)

    for sub in sub_list:
        if ('00' in sub)|('C' in sub):
            dir = augm_data_path
        else:
            dir = data_path
        images_per_sub = dict()
        images_per_sub['image'] = []
        if mask:
            images_per_sub['mask'] = []
            mask_path = os.path.join(data_path, f'sub-{sub}', f'sub-{sub}_t1_brain-final_mask.nii.gz')    
            if os.path.isfile(mask_path):
                images_per_sub['mask'] = mask_path
            else:
                missing_files['mask'].append(mask_path)
        
        for feat in feat_params:
            proposed_map_paths = assign_feature_maps(sub, feat, dir, fold=fold)
            map_path = None # path of the `feat`
                
            # in case `proposed_map_paths` is single path
            if not isinstance(proposed_map_paths, list):
                proposed_map_paths = [proposed_map_paths]
            
            for proposed_map_path in proposed_map_paths:
                if os.path.isfile(proposed_map_path):
                    map_path = proposed_map_path
            
            if map_path is not None:
                # feature path found and added to `image` field 
                images_per_sub['image'].append(map_path)
            else:
                missing_files['image'].append(proposed_map_path)
        if ('00' in sub)|('C' in sub):
            seg_path = os.path.join(augm_data_path, f'split{fold}/masks/sub-{sub}_t1_brain-final.nii.gz')
        else:
            seg_path = os.path.join(label_path, f'{sub}.nii.gz')
        
        if os.path.isfile(seg_path):
            images_per_sub['seg'] = seg_path
        else:
            missing_files['seg'].append(seg_path)
            
        files.append(images_per_sub)
    print(missing_files)    
    return files, missing_files



def scaling_specified(data_dict, features, scaling_dict):
    '''
    features - list of features e.g. ['image', 'curv', 'sulc',...]
    scaling_dict - {
                    'feature_name_1': [a, b], - use provided `a` and `b` for (x-a)/b normalization
                    'feature_name_2': None, - infer `a_min` and `a_max` from the data for min-max normalization
                   }
    '''
    mask_bool = data_dict["mask"][0] > 0.
    for i, feature in enumerate(features):
        v = scaling_dict[feature]
        data_dict["image"][i][mask_bool] = normalize_(data_dict["image"][i][mask_bool], ab=v)
    return data_dict

def scaling_specified_wrapper(features, scaling_dict):
    '''
    decorate `minmax_scaling_specified` with pre-specified `scaling_dict`
    '''
    def wrapper(data_dict):
        return scaling_specified(data_dict, features=features, scaling_dict=scaling_dict)
    return wrapper

def scaling_as_torchio(data_dict, features, scaling_dict):
    mask_bool = data_dict["mask"] > 0.
    features_ = features
    for i, feature in enumerate(features_):
        #  condition, beceause some features like curv, sulc, thickness - don't need in scale, however, can be done.
        landmarks_path = Path(f'./landmarks/{feature}_landmarks.npy')
        landmark =  np.load(landmarks_path)
        d = data_dict["image"][i].clone().detach()
        # d = data_dict["image"][i].copy()
        m = mask_bool.clone().detach()
        # m = mask_bool.copy()
        d_n = histogram_standardization._normalize(d, landmark, m)
        tensor = z_normalization.ZNormalization.znorm(d_n, m)
        if tensor is not None:
            data_dict["image"][i] = tensor
            #print(f'{feature} normalized: \n Min Value: {data_dict["image"][i].min()} \n Max Value: {data_dict["image"][i].max()}')
    return data_dict

def scaling_torchio_wrapper(features, scaling_dict):
    '''
    decorate `minmax_scaling_specified` with pre-specified `scaling_dict`
    '''
    def wrapper(data_dict):
        return scaling_as_torchio(data_dict, features=features, scaling_dict=scaling_dict)
    return wrapper
    
def binarize_target(data_dict, eps=1e-3):
    '''
    data_dict - [C,H,W,D]
    '''
    data_dict["seg"] = (data_dict["seg"] > eps).to(data_dict["seg"].dtype) #.astype(data_dict["seg"].dtype)
    return data_dict
    
    
def mask_transform(data_dict):
    '''
    data_dict - [C,H,W,D]
    '''
    data_dict["mask"] = (data_dict["mask"] > 0).astype(data_dict["image"].dtype)
    data_dict["image"] = data_dict["image"] * (data_dict["mask"])
    return data_dict

def setup_transformations(config, features, scaling_dict=None):
    
    interpolate = config.default.interpolate
    if interpolate:
        spatial_size_conf = tuple(config.default.interpolation_size)
        
    assert config.dataset.trim_background
    keys=["image", "seg", "mask"]
    sep_k=["seg", "mask"]
    
    if scaling_dict in 'torchio':
        scaler = scaling_torchio_wrapper(features, scaling_dict)
        print(scaling_dict)
    elif scaling_dict in 'scale_metadata':
        scaler = scaling_specified_wrapper(features, scaling_dict)
    else:
        scaler = ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True)
    print(spatial_size_conf)  
    # no-augmentation transformation
    val_transf = Compose(
                        [
                            LoadImaged(keys=keys),
                            EnsureChannelFirstd(keys=keys)
                        ] + ([Resized(keys=keys, spatial_size=spatial_size_conf)] if interpolate else []) + \
                        [
                            # Spacingd(keys=sep_k, pixdim=1.0),
                            mask_transform,
                      ToTensord(keys=keys, dtype=torch.float), # zero the non-mask values
                            binarize_target,
                            scaler,
                            EnsureTyped(keys=sep_k, dtype=torch.float),
                        ]
                        )
        
    if config.opt.augmentation:
        
        rand_affine_prob = config.opt.rand_affine_prob
        rot_range = config.opt.rotation_range
        shear_range = config.opt.shear_range
        scale_range = config.opt.scale_range
        translate_range = config.opt.translate_range

        noise_std = config.opt.noise_std
        flip_prob = config.opt.flip_prob
        rand_zoom_prob = config.opt.rand_zoom_prob
        
        # basic operations
        transforms = [LoadImaged(keys=keys), 
                      EnsureChannelFirstd(keys=keys),
                      
                     ] + ([Resized(keys=keys, spatial_size=spatial_size_conf)] if interpolate else []) + \
                     [mask_transform,
                      ToTensord(keys=keys, dtype=torch.float),
                      binarize_target,scaler]#, Spacingd(keys=sep_k, pixdim=1.0)]
        
        if rand_affine_prob == 0 and rot_range > 0:
            transforms.append(RandRotated(keys=keys, # apply to all!
                                range_x=rot_range, 
                                range_y=rot_range, 
                                range_z=rot_range, 
                                prob=0.5)
                             )
        if flip_prob > 0:
            transforms.append(RandFlipd(keys=keys, # apply to all!
                                        prob=flip_prob, 
                                        spatial_axis=0))
            
        if rand_affine_prob > 0:
            transforms.append(RandAffined(prob=rand_affine_prob, 
                                         rotate_range=[rot_range, rot_range, rot_range], 
                                         shear_range=[shear_range, shear_range, shear_range], 
                                         translate_range=[translate_range, translate_range, translate_range], 
                                         scale_range=[scale_range, scale_range, scale_range], 
                                         padding_mode='zeros',
                                         keys=keys # apply to all!
                                        )
                             )

        if noise_std > 0:
            transforms.append(RandGaussianNoised(prob=0.5, 
                                                mean=0.0, 
                                                std=noise_std, 
                                                keys=["image"]
                                               )
                             )
                              
        if rand_zoom_prob > 0:
            transforms.append(RandZoomd(prob=0.5, min_zoom=0.9, max_zoom=1.1, keys=keys))
        
        # add the rest 
        transforms.extend([ # zero the non-mask values
                            EnsureTyped(keys=sep_k, dtype=torch.float),
                         ]
                        )
        
        train_transf = Compose(transforms)
    else:
        train_transf = val_transf
    
    return train_transf, val_transf


def setup_dataloaders(config, data_path, label_path, augm_data_path, feat_params, fold):
    metadata_path = config.dataset.metadata_path
    
    scaling_dict = None
    if config.dataset.scaling_method in 'torchio':
        scaling_dict = 'torchio'
    elif config.dataset.scaling_method in 'scale_metadata':
        
        scaling_data_path = config.dataset.scaling_metadata_path
        scaling_dict = np.load(scaling_data_path, allow_pickle=True).item()
    else:
        print('Warning! no SCALING METADATA used! Applying naive independent MinMax...')
    
    split_dict = np.load(metadata_path, allow_pickle=True)
    train_list = split_dict[fold].get('train')
    val_list = split_dict[fold].get('val')
    
    # Flag to add mask as additional sequence to Subset
    add_mask = config.dataset.trim_background
    
    train_files, train_missing_files = create_datafile(train_list, feat_params, data_path, label_path, augm_data_path, mask=add_mask, fold=fold)
    val_files, val_missing_files = create_datafile(val_list, feat_params, data_path, label_path, augm_data_path, mask=add_mask, fold=fold)
    
    print(scaling_dict)
    train_transf, val_transf = setup_transformations(config, feat_params, scaling_dict)
    
    # training dataset
    train_ds = monai.data.Dataset(data=train_files, transform=train_transf)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.opt.train_batch_size,
        shuffle=config.dataset.shuffle_train,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # validation dataset
    val_ds = monai.data.Dataset(data=val_files, transform=val_transf)
    val_loader = DataLoader(val_ds, 
                            batch_size=config.opt.val_batch_size, 
                            num_workers=0, 
                            collate_fn=list_data_collate,
                            shuffle=False # important not to shuffle, to ensure label correspondence
                            )
    return train_loader, val_loader