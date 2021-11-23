import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

# COnvert to torch tensor
def to_variable(tensor, volatile=False, requires_grad=True):
    return Variable(tensor.float().cuda(), requires_grad=requires_grad)

# Initialize model weights
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def initialize_weights_new(model, manual_seed=7):
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def make_patches(x, patch_size):
    if x.dim()>3:
        channel_dim = x.shape[1]
        patches = x.unfold(1, channel_dim, channel_dim).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(-1, channel_dim, patch_size, patch_size)
    else:
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(-1, patch_size, patch_size)
    return patches, unfold_shape

