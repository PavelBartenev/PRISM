## PRISM: Pathological Region Inpainting for Scarce MRI data

This is code for PRISM paper. Here we propose a method for generating synthetic pathological MRI data via diffusion inpainting. 

### Requirements

In this repo you can find two models for synthetic pathological MRI generation: I2SB and 3D Conditional Diffusion. 
We strongly encourage you to use separate environments for work with these models as described below.

#### 2D Conditional Diffusion

```
cd 2d_conditional_diffusion
conda env create --file requirements.yaml
conda activate diffusion2d
```

#### 3D Conditional Diffusion

```
cd 3d_conditional_diffusion
conda create -n diffusion3d python=3.10.0
conda activate diffusion3d
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
```

### Experiments

#### I2SB

To run the training of the model:

```
python train.py --name $EXPERIMENT_NAME --dataset-dir $DATA_DIR --batch-size 4 --microbatch 2 --log-dir $LOG_DIR --split-id $SPLIT_ID --splits-filename $SPLITS_FILENAME --gpu-id 0 --num-itr 29400 --save-model-iter 29399 --corrupt inpaint
```

Before running the generation you need to obtain masks for synthetic pathologies. This can be done using Jupyter notebooks from 3DUnet-masks folder. 
To run the generation:

```
python sample.py --ckpt $CKPT_NAME --batch-size 4 --mri-path $HEALTHY_DATA_DIR --mask-path $MASKS_DIR --patch-mask-path $PATCH_MASK_PATH --gpu-id 0
```

To incorporate patches with generated pathology into MRI you should use the Jupyter notebook from notebooks subfolder.


#### 3D Conditional Diffusion

To run the training of the model:

```
python3 train_inpainting.py --fcd_patches_dataset --with_condition --loss_type l1_masked --inputfolder $DATA_PATH --results_folder $RESULTS_PATH --splits_filename $SPLITS_FILENAME --split_id $SPLIT_ID --batchsize 1 --epochs 500001 --train_lr 1e-5 --input_size 40 --depth_size 40 --num_channels 64 --channel_mult 1,2,4,4 --num_res_blocks 1 --timesteps 250 --save_and_sample_every 250000
```

Before running the generation you need to obtain masks for synthetic pathologies. This can be done using Jupyter notebooks from 3DUnet-masks folder. 

To run the generation:

```
python3 sample_inpainting.py --ckpt_path $CKPT_NAME --mri_dir $HEALTHY_DATA_DIR --mask_dir $MASKS_DIR --patch_mask_dir $PATCH_MASK_PATH --save_dir $SAVE_DIR
```

#### Checkpoints
Checkpoints of trained models can be found by the [link](https://drive.google.com/drive/folders/1_rlLW1DtvZLv4e6n52zJJifvrmciPv8Q?usp=drive_link)




















