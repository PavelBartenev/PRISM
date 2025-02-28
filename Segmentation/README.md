# SegmentationNet

This repository contains scripts for training and inference of a neural network model for medical image segmentation using the V2V model architecture. The scripts are designed to handle data preparation, model training, and evaluation of segmentation metrics.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

## Requirements

- Python 3.7 or higher
- PyTorch
- MONAI
- NumPy
- Matplotlib
- TQDM
- TensorBoardX
- Nibabel
- Nilearn

## Setup

1. **Clone the repository:**

2. **Install the required packages:**

   You can install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```
 MONAI_DATA_DIRECTORY=./MONAI_TMP
   ```

## Data Preparation

Ensure that your data is organized as follows:

- `data_path`: Directory containing the input data files.
The `data_path` should contain subdirectories for each subject, with each subdirectory containing the brain image and its corresponding mask:

```
data_path/
└── sub-{sub_ind}/
    ├── sub-{sub_ind}_t1_brain-final.nii.gz
    └── sub-{sub_ind}_t1_brain-final_mask.nii.gz
...
```
- `label_path`: Directory containing the ground truth label files.
The `label_path` should contain label files for each subject:

```
label_path/
└── {sub_ind}.nii.gz
```

- `augm_data_path`: Directory containing the augmented data files.
The `augm_data_path` should be organized into splits, each containing `masks` and `t1` directories:
augm_data_path/
└── split{i}/
    ├── masks/
    │   ├── sub-{sub_ind}_t1_brain-final.nii.gz
    └── t1/
        └── sub-{sub_ind}_t1_brain-final.nii.gz

Also your directory should have a `metadata/stratified.npy` file with stratification into folds for training and inference. The structure of the file is list of dicts with 'train' and 'val' keys. Name your generated subjects starting with 'C'. 


## Training

To train the model, use the `train.py` script. The script requires the following arguments:

- `--dir`: Path to the dataset directory (`data_path`).
- `--label_dir`: Path to the directory containing label masks (`label_path`).
- `--augm_dir`: Path to the directory containing augmented data (`augm_data_path`).
- `--mod`: Modality (default: `t1`).
- `--fold`: Fold index for cross-validation (default: `0`).

Example command:

```bash
python train.py --dir /path/to/data --label_dir /path/to/labels --augm_dir /path/to/augmented --mod t1 --fold 0
```

## Inference

To perform inference and calculate metrics, use the `inference.py` script. The script requires the following arguments:

- `--data_path`: Path to the data files.
- `--label_path`: Path to the label files.
- `--augm_data_path`: Path to the augmented data files.
- `--logdir`: Directory containing the model weights. 
- `--fold`: Fold index for cross-validation.

Example command:

```bash
python inference.py --data_path /path/to/data --label_path /path/to/labels --augm_data_path /path/to/augmented --logdir /path/to/logdir --fold 0
```

## Results

- Training logs and model checkpoints will be saved in the specified log directory.
- Inference results, including calculated metrics, will be printed to the console.



For further assistance, please open an issue on the repository or contact the maintainers.
