import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st

def _get_kernel(kernlen=16, std=3):
    interval = (2*std+1.)/kernlen
    x = np.linspace(-std-interval/2., std+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel #(2D Gaussian Kernel)

def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8) #(Normalizing pixel values)

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        gaussian_kernel = np.float32(_get_kernel(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))
    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))   
        return x