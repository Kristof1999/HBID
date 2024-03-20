# based on Levin et al. work: 
# Efficient Marginal Likelihood Optimization in Blind Deconvolution

import torch
from torch import nn
import torchvision.transforms.functional as TF

class LevinLossYPred(nn.Module):
    def __init__(self):
        super(LevinLossYPred, self).__init__()
    
    def forward(self, y_estimate, y, covariance, k, likelihood_reg=1):
        mse = nn.MSELoss(reduction='mean')
        
        k_sz1 = k.shape[-2]
        k_sz2 = k.shape[-1]

        k = k.squeeze()
        y = y.squeeze()
        k = k[None, None, :, :]
        y = y[None, None, :, :]

        centered = torch.fft.fftshift(covariance, dim=(-2, -1))
        ssd2 = TF.center_crop(centered, (k_sz1*2-1, k_sz2*2-1))
        ssd2 = ssd2.squeeze()
        ssd2 = ssd2[None, None, :, :]
        
        conv2 = torch.nn.functional.conv2d(ssd2, k.rot90(k=2, dims=(-2,-1)))
        conv2 = torch.nn.functional.conv2d(conv2, k).squeeze()

        return likelihood_reg * mse(y_estimate, y) + conv2