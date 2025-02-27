## PRISM: Pathological Region Inpainting for Scarce MRI data

This is code for PRISM paper. Here we propose a method for generating synthetic pathological MRI data via diffusion inpainting. 

### Requirements

In this repo you can find two models for synthetic pathological MRI generation: I2SB and 3D Conditional Diffusion. 
We strongly encourage you to use separate environments for work with these models as described below.

#### I2SB

```
cd I2SB
conda env create --file requirements.yaml
conda activate i2sb
```

#### 3D Conditional Diffusion

```
cd med-ddpm-main
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
python sample.py --ckpt $CKPT_NAME --batch-size 4 --mri-path $HEALTHY_DATA_DIR --mask-path $MASKS_DIR --patch-mask-path $MASKS_PATCHES_PATH --gpu-id 0
```


#### 3D Conditional Diffusion




















