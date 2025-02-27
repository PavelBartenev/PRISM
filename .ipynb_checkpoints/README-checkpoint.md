## PRISM: Pathological Region Inpainting for Scarce MRI data

This is code for PRISM paper. Here we propose a method for generating synthetic pathological MRI data via diffusion inpainting. 

### Requirements

In this repo you can find two models for synthetic pathological MRI generation: I2SB and 3D Conditional Diffusion. 
We strongly encourage you to use separate environments for work with these models as described below.

#### I2SB

cd I2SB
conda env create --file requirements.yaml
conda activate i2sb

#### 3D Conditional Diffusion

cd med-ddpm-main
conda create -n diffusion3d python=3.10.0
conda activate diffusion3d
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116


### Experiments

