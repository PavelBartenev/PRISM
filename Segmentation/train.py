import os
import time
import shutil
import tempfile
import argparse
import traceback
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast
from torchinfo import summary
from tensorboardX import SummaryWriter
from monai.config import print_config
from monai.utils import set_determinism
import configdot


from v2v import V2VModel
from losses import *
from dataset import setup_dataloaders
from utils import save, get_capacity, calc_gradient_norm, get_label, show_prediction_slice


# Print MONAI configuration
print_config()

# Define the main training function
def main(data_path, label_path, augm_data_path, mod, fold):
    config = configdot.parse_config('config-cv.ini')
    device = setup_device(config)
    feat_params = config.dataset.features
    experiment_name, writer = setup_experiment_logging(config, augm_data_path, mod, fold)
    model = setup_model(config, device)
    criterion, opt = setup_optimization(config, model)
    train_loader, val_loader = setup_dataloaders(config, data_path, label_path, augm_data_path, feat_params, fold)
    run_training(config, model, criterion, opt, train_loader, val_loader, device, writer, experiment_name, fold)

# Setup device for training
def setup_device(config):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{config.opt.device}')
        print('Using GPU#:', torch.cuda.current_device())
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(device)
    return device

# Setup experiment logging
def setup_experiment_logging(config, augm_data_path, mod, fold):
    if augm_data_path is 'none':
        experiment_name = f'v2v-{mod}_original@{datetime.now().strftime("%d.%m.%Y-%H")}@fold-{fold}'
    else:
        gen_model = augm_data_path.split('/')[-2]
        mask_type = augm_data_path.split('/')[-1]
        experiment_name = f'v2v-{mod}_augm_{gen_model}_{mask_type}@{datetime.now().strftime("%d.%m.%Y-%H")}@fold-{fold}'
    print("Experiment name: {}".format(experiment_name))

    writer = None
    if config.default.make_logs:
        experiment_dir = os.path.join(config.default.log_dir, experiment_name)
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)
        shutil.copy('config-cv.ini', os.path.join(experiment_dir, "config-cv.ini"))

        if config.dataset.save_best_val_predictions:
            val_preds_path = os.path.join(experiment_dir, 'best_val_preds')
            config.dataset.val_preds_path = val_preds_path
            os.makedirs(val_preds_path)

        writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    return experiment_name, writer

# Setup model based on configuration
def setup_model(config, device):
    if config.model.name == "v2v":
        model = V2VModel(config).to(device)
    summary(model)
    capacity = get_capacity(model)
    print(f'Model created! Capacity: {capacity}')

    if hasattr(config.model, 'weights'):
        model_dict = torch.load(config.model.weights, map_location='cpu')
        print(f'LOADING from {config.model.weights} \n epoch:', model_dict['epoch'])
        model.load_state_dict(model_dict['model_state'])

    return model

# Setup optimization criteria and optimizer
def setup_optimization(config, model):
    criterion = {
        "Dice": dice_loss_custom,
        }[config.opt.criterion]

    opt = optim.Adam(model.parameters(), lr=config.opt.lr)
    return criterion, opt

# Run the training process
def run_training(config, model, criterion, opt, train_loader, val_loader, device, writer, experiment_name, fold):
    metric_dict_epoch = defaultdict(list)
    n_iters_total_train = 0
    n_iters_total_val = 0
    target_metric_prev = -1

    try:
        for epoch in range(config.opt.start_epoch, config.opt.n_epochs):
            
            print (f'TRAIN EPOCH: {epoch} ... ')
            n_iters_total_train, _  = one_epoch(model, 
                                            criterion, 
                                            opt, 
                                            config, 
                                            train_loader, 
                                            device, 
                                            writer, 
                                            epoch, 
                                            metric_dict_epoch, 
                                            n_iters_total_train,
                                            is_train=True)
            
            print (f'VAL EPOCH: {epoch} ... ')
            n_iters_total_val, target_metric = one_epoch(model, 
                                            criterion, 
                                            opt, 
                                            config, 
                                            val_loader, 
                                            device, 
                                            writer, 
                                            epoch, 
                                            metric_dict_epoch, 
                                            n_iters_total_val,
                                            is_train=False)

            if config.opt.save_model and config.default.make_logs:
                if not config.model.use_greedy_saving:
                    print(f'SAVING...')
                    save(experiment_name, model, opt, epoch)
                elif target_metric > target_metric_prev:
                    print(f'target_metric = {target_metric}, SAVING...')
                    save(experiment_name, model, opt, epoch)
                    target_metric_prev = target_metric
    except Exception as e:
        print(traceback.format_exc())

# Run the training/validation process on 
def one_epoch(model, 
              criterion, 
              opt,
              config, 
              dataloader, 
              device, 
              writer, 
              epoch, 
              metric_dict_epoch, 
              n_iters_total=0,
              is_train=True):

    # use amp to accelerate training
    if config.opt.use_scaler:
        scaler = torch.cuda.amp.GradScaler()

    phase_name = 'train' if is_train else 'val'
    loss_name = config.opt.criterion
    metric_dict = defaultdict(list)
    target_metric_name = config.model.target_metric_name 

    if not is_train:
        model.eval()
    else:
        model.train()

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = enumerate(dataloader)
        val_predictions = {}
        for iter_i, data_tensors in iterator:
            brain_tensor, label_tensor, mask_tensor = (data_tensors['image'].to(device),
                                                       data_tensors['seg'].to(device),
                                                       data_tensors['mask'].to(device))
            # forward pass
            t1 = time.time()
            with autocast(enabled=config.opt.use_scaler):
                label_tensor_predicted = model(brain_tensor) # -> [bs,1,ps,ps,ps]
                loss = criterion(label_tensor_predicted, label_tensor) 

            if is_train:
                opt.zero_grad()

                if config.opt.use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if hasattr(config.opt, "grad_clip"):
                    if config.opt.use_scaler:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           config.opt.grad_clip)

                metric_dict['grad_norm'].append(calc_gradient_norm(filter(lambda x: x[1].requires_grad, 
                                                model.named_parameters())))

                if config.opt.use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

            t2 = time.time()    
            dt = t2-t1 # inference time
            
            metric_dict[f'batch_time'].append(dt)
            metric_dict[f'{loss_name}'].append(loss.item())
            label_tensor_predicted = label_tensor_predicted * mask_tensor
            
            if hasattr(config.opt, 'save_inference') and config.opt.save_inference:
                show_prediction_slice(iter_i, 
                                      brain_tensor, 
                                      mask_tensor, 
                                      label_tensor, 
                                      label_tensor_predicted, 
                                      b_ind=0, 
                                      c_ind=0)
            
            cov = coverage(label_tensor_predicted, label_tensor).item()
            fp = false_positive(label_tensor_predicted, label_tensor).item()
            fn = false_negative(label_tensor_predicted, label_tensor).item()
            dice = dice_score(label_tensor_predicted, label_tensor).item()
            
            if not is_train and config.dataset.save_best_val_predictions:
                label = get_label(dataloader.dataset.data[iter_i]['seg'])
                val_predictions[label] = label_tensor_predicted.detach().cpu().numpy()
            
            metric_dict['coverage'].append(cov) # a.k.a recall
            metric_dict['false_positive'].append(fp)
            metric_dict['false_negative'].append(fn)
            metric_dict['dice_score'].append(dice)
            
            #########
            # PRINT #
            #########
            message = f'For {phase_name}, iter: {iter_i},'
            for title, value in metric_dict.items():
                if title == 'grad_norm':
                    v = np.round(value[-1],6)
                else:
                    v = np.round(value[-1],3)
                message+=f' {title}:{v}'
            print(message)


            if is_train and writer is not None:
                for title, value in metric_dict.items():
                    writer.add_scalar(f"{phase_name}_{title}", value[-1], n_iters_total)

            n_iters_total += 1

    target_metric = 0
    for title, value in metric_dict.items():
        m = np.mean(value)
        metric_dict_epoch[phase_name + '_' + title].append(m)
        if title == target_metric_name:
            target_metric = m
        if writer is not None:
            writer.add_scalar(f"{phase_name}_{title}_epoch", m, epoch)
            
    #####################
    # SAVING BEST PREDS #
    #####################
    target_metrics_epoch = metric_dict_epoch[f'val_{target_metric_name}']
    if not is_train:
        if config.dataset.save_best_val_predictions:
            # use greedy-saving: save only if the target metric is improved
            if len(target_metrics_epoch) == 1 or target_metrics_epoch[-1] >= target_metrics_epoch[-2]:
                for label, pred in val_predictions.items():
                    torch.save(pred, os.path.join(config.dataset.val_preds_path, f'{label}'))

    return n_iters_total, target_metric


# Main entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FCDNet')
    parser.add_argument('-d', '--dir', help='Path to our dataset', required=True)
    parser.add_argument('-ld', '--label_dir', help='Path to masks for our dataset', required=True)
    parser.add_argument('-ad', '--augm_dir', help='Path to generated brains', required=True)
    parser.add_argument('-m', '--mod', help='Workflow directory', default='t1', required=False)
    parser.add_argument('-f', '--fold', help='Fold', default='0', required=False)
    args = parser.parse_args()

    data_path = args.dir
    label_path = args.label_dir
    augm_data_path = args.augm_dir
    mod = args.mod
    fold = int(args.fold)

    set_determinism(seed=42)
    os.makedirs('./MONAI_TMP', exist_ok=True)
    os.environ['MONAI_DATA_DIRECTORY'] = "./MONAI_TMP"
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory

    start = time.time()
    main(data_path, label_path, augm_data_path, mod, fold)
    print('Complete')
    end = time.time()
    print('total time (s)= ' + str(end-start))
