# https://github.com/frankkramer-lab/MIScnn
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def calculate_metrics(data_resica, data_active, n=15):
    """
    Calculates matrics as functions of the threshold
    Returns: Maximum metrics and intesity, where maximum sum of metrics
    
    """
    # data_resica - pred
    # data_active - gt
    metrics = []
    if (data_resica==0).all():
        return 0,0,1,0,0,0
    
    for thresh in np.linspace(0,data_resica.max(),n):
        tn, fp, fn, tp = confusion_matrix((data_active > 0.5).reshape(-1), (data_resica > thresh).reshape(-1), labels=[0, 1]).ravel()
        large_enough = int((np.where((data_resica > thresh),1,0)).sum() > 800)
        metrics.append([
                        thresh,
                        large_enough*tp/(tp+fp),
                        large_enough*tp/(tp+fn),
                        large_enough*tn/(tn+fp),
                        large_enough*2*tp/(2*tp+fp+fn),
                        large_enough*(tp+tn)/(tp+tn+fp+fn),
                        large_enough*tp/(data_resica > thresh).reshape(-1).size])
        
    metrics = np.nan_to_num(np.array(metrics))
    
    ind = np.argmax(metrics[:,1])#+metrics[:,2]+metrics[:,4])
    intensity =metrics[:,0][ind]
    Precision = metrics[ind,1]
    Sensitivity = metrics[ind,2]
    Specificity = metrics[ind,3]
    Dice = metrics[ind,4]
    Accuracy = metrics[ind,5]
    #Accuracy_tp = metrics[ind,6]
    
    return Precision,Sensitivity,Specificity,Dice,intensity,Accuracy


def coverage(input, target):
    assert input.shape[1] == target.shape[1] 
    assert input.shape[1] == 1
    intersection = (input*target).sum(dim=(-1,-2,-3)) 
    cardinality = target.sum(dim=(-1,-2,-3)) + 1e-10
    return (intersection/cardinality).mean()

def false_positive(input, target):
    assert input.shape[1] == target.shape[1] 
    assert input.shape[1] == 1
    intersection = (input*(1-target)).sum(dim=(-1,-2,-3)) 
    cardinality = (1-target).sum(dim=(-1,-2,-3))
    return (intersection/cardinality).mean()

def false_negative(input, target):
    assert input.shape[1] == target.shape[1] 
    assert input.shape[1] == 1
    intersection = ((1-input)*target).sum(dim=(-1,-2,-3)) 
    cardinality = (target).sum(dim=(-1,-2,-3))
    return (intersection/cardinality).mean()

def dice_score(input, target):
    '''
    Binary Dice score
    input - [batch_size,1,H,W,D], probability [0,1]
    target - binary mask [batch_size,1,H,W,D], 1 for foreground, 0 for background

    Dice = 2TP/(2TP + FP + FN) = 2X*Y/(|X| + |Y|)
    '''

    target = target.squeeze(1) # squeeze channel 
    input = input.squeeze(1) # squeeze channel
    epsilon = 1e-10

    intersection = 2*torch.sum(input * target, dim=(1,2,3)) + epsilon # [bs,]
    cardinality = torch.sum(input, dim=(1,2,3)) + torch.sum(target, dim=(1,2,3)) + epsilon # [bs,]
    dice_score = intersection / cardinality

    return dice_score.mean()


def dice_loss_custom(input, target):
    '''
    Binary Dice loss
    input - [batch_size,1,H,W,D], probability [0,1]
    target - binary mask [batch_size,1,H,W,D], 1 for foreground, 0 for background
    '''

    target = target.squeeze(1) # squeeze channel 
    input = input.squeeze(1) # squeeze channel
    
    intersection = 2*torch.sum(input * target, dim=(1,2,3)) + 1 # [bs,]
    cardinality = torch.sum(torch.pow(input,2) + torch.pow(target,2), dim=(1,2,3)) + 1 # [bs,]
    dice_score = intersection / cardinality

    return 1-dice_score.mean()

