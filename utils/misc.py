'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def confidence_filter(x, predictions, task_index, image_index, threshold_s=0.95):
    if task_index == 0:
        student_prediction = predictions[0][image_index].unsqueeze(0)
        x_prob, x_pseudo_labels = F.softmax(x, dim=1).max(1)
        binary_mask = (x_prob > threshold_s).type(torch.FloatTensor).cuda()
        #confidence_map = x_pseudo_labels * binary_mask
        return x_pseudo_labels, x_prob, binary_mask       
    elif task_index == 1:
        student_prediction = predictions[1][image_index].unsqueeze(0)
        depth_diff = torch.abs(x - student_prediction)
        confidence_map = torch.clamp(1 - depth_diff, min=0.1, max=0.9)
        x_prob = 0
        return confidence_map, x_prob, x_prob
    elif task_index == 2:
        student_prediction = predictions[2][image_index].unsqueeze(0)
        teacher_normals = F.normalize(x, p=2, dim=1)
        student_normals = F.normalize(student_prediction, p=2, dim=1)
        # Compute Cosine Sim
        cosine_similarity = F.cosine_similarity(teacher_normals, student_normals, dim=1)  # Shape: [1, 288, 384]
        # Normalise to 0,1
        confidence_map = (cosine_similarity + 1) / 2
        # confidence map
        confidence_map = torch.clamp(confidence_map, min=0.1, max=0.9)
        x_prob = 0
        return confidence_map, x_prob, x_prob

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def softmax_mse_loss(input, target):
    """
    Compute the softmax cross entropy loss
    Args:
        input (torch.Tensor): input
        target (torch.Tensor): target
    Returns:
        torch.Tensor: softmax cross entropy loss
    """
    return torch.mean(torch.sum(F.softmax(torch.exp(input), dim=1) * F.mse_loss(input, target, reduction='none'), dim=1))

def get_current_consistency_weight(epoch):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    consistency = 10.0
    consistency_rampup = 5.0
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def dynamic_thresholding(logits, q_base=0.5, alpha=0.8, q_min=0.5, q_max=0.99):
    # Hyperparameters
    #q_base = 0.9  # Base quantile
    #alpha = 0.3   # Sensitivity factor
    #q_min, q_max = 0.7, 0.95  # Clamp range for thresholds

    # Flatten the spatial dimensions for each class
    probs_flat = logits.view(1, 13, -1)  # Shape: [1, 13, 288*384]
    
    # Calculate mean confidence for each class
    class_means = probs_flat.mean(dim=-1)  # Shape: [1, 13]

    # Dynamic quantile calculation
    dynamic_quantiles = q_base + alpha * (0.5 - class_means)  # Shape: [1, 13]
    dynamic_quantiles = torch.clamp(dynamic_quantiles, q_min, q_max)  # Clamp to bounds

    # Compute thresholds for each class
    thresholds = torch.empty_like(class_means)
    for c in range(probs_flat.size(1)):  # Iterate over classes
        thresholds[:, c] = torch.quantile(probs_flat[:, c, :], dynamic_quantiles[:, c].item(), dim=-1)

    # Generate binary masks based on thresholds
    binary_masks = probs_flat > thresholds.unsqueeze(-1)  # Shape: [1, 13, 288*384]
    binary_masks = binary_masks.view_as(logits)  # Reshape back to original spatial dimensions    

    combined_mask = binary_masks.any(dim=1, keepdim=True)  # Shape: [B, 1, H, W]
    return combined_mask
