import argparse
from collections import defaultdict
from statistics import mean

import numpy as np
import torch
from tqdm import tqdm
import monai
from monai.data import DataLoader, list_data_collate

from losses import calculate_metrics, coverage
import configdot
from dataset import  create_datafile, setup_transformations
from utils import get_label, get_latest_weights
from v2v import V2VModel

# Define the main function for metrics
def metrics_all(data_path, label_path, augm_data_path, logdir, i):
    config = configdot.parse_config('config-cv.ini')
    assert config.opt.val_batch_size == 1
    device = torch.device('cpu')

    # Load model
    best_model = load_model(config, logdir, device)

    # Setup data
    val_loader = setup_validation_data(data_path, label_path, augm_data_path, config, i)

    # Initialize metrics
    metric_dict = defaultdict(list)

    # Evaluate model
    evaluate_model(best_model, val_loader, metric_dict, device)

    # Calculate and print mean metrics
    Precision, Sensitivity, Specificity, Dice, Coverage = calculate_and_print_mean_metrics(metric_dict)

    return Precision, Sensitivity, Specificity, Dice, Coverage

# Load the model
def load_model(config, logdir, device):
    assert config.model.name == "v2v"
    best_model = V2VModel(config).to(device)
    model_dict = torch.load(get_latest_weights(logdir), map_location=device)
    best_model.load_state_dict(model_dict['model_state'])
    best_model.eval()
    return best_model

# Setup validation data
def setup_validation_data(data_path, label_path, augm_data_path, config, i):
    metadata_path = config.dataset.metadata_path
    split_dict = np.load(metadata_path, allow_pickle=True)
    val_list = split_dict[i].get('val')
    feat_params = config.dataset.features
    add_mask = config.dataset.trim_background
    val_files, _ = create_datafile(val_list, feat_params, data_path, label_path, augm_data_path, mask=add_mask)
    _, val_transf = setup_transformations(config, feat_params, scaling_dict='torchio')
    val_ds = monai.data.Dataset(data=val_files, transform=val_transf)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate, shuffle=False)
    return val_loader

# Evaluate the model
def evaluate_model(best_model, val_loader, metric_dict, device):
    with torch.no_grad():
        for iter_i, data_tensors in tqdm(enumerate(val_loader)):
            brain_tensor, label_tensor, mask_tensor = (
                data_tensors['image'].to(device),
                data_tensors['seg'].to(device),
                data_tensors['mask'].to(device)
            )

            label = get_label(val_loader.dataset.data[iter_i]['seg'])
            print(f'Label: {label}')

            # Forward pass
            label_tensor_predicted = best_model(brain_tensor) * mask_tensor
            brain = brain_tensor[0, 0].detach().cpu().numpy()
            label_pred_arr = label_tensor_predicted[0, 0].detach().cpu().numpy()
            label_gt_arr = label_tensor[0, 0].detach().cpu().numpy()

            # Calculate metrics
            Precision, Sensitivity, Specificity, Dice, _, _ = calculate_metrics(label_pred_arr, label_gt_arr)
            print(f'Dice {round(Dice, 3)}\n', f'Precision {round(Precision, 3)}\n', f'Sensitivity {round(Sensitivity, 3)}\n', f'Specificity {round(Specificity, 3)}')
            Coverage = coverage(label_tensor, label_tensor_predicted).item()
            metric_dict[label] = [Precision, Sensitivity, Specificity, Dice, Coverage]

# Calculate and print mean metrics
def calculate_and_print_mean_metrics(metric_dict):
    Precision = [metric[0] for metric in metric_dict.values()]
    Sensitivity = [metric[1] for metric in metric_dict.values()]
    Specificity = [metric[2] for metric in metric_dict.values()]
    Dice = [metric[3] for metric in metric_dict.values()]
    Coverage = [metric[4] for metric in metric_dict.values()]

    print(f'Precision mean on validation: {mean(Precision)}')
    print(f'Sensitivity mean on validation: {mean(Sensitivity)}')
    print(f'Specificity mean on validation: {mean(Specificity)}')
    print(f'Dice on validation: {mean(Dice)}')
    print(f'Coverage mean on validation: {mean(Coverage)}')
    return Precision, Sensitivity, Specificity, Dice, Coverage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for calculating metrics.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data files.')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label files.')
    parser.add_argument('--augm_data_path', type=str, required=True, help='Path to the augmented data files.')
    parser.add_argument('--logdir', type=str, required=True, help='Directory containing the model weights.')
    parser.add_argument('--fold', type=int, required=True, help='Fold index for cross-validation.')

    args = parser.parse_args()

    Precision, Sensitivity, Specificity, Dice, Coverage = metrics_all(args.data_path, args.label_path, args.augm_data_path, args.logdir, args.fold)

