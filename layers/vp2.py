# based on VPNet: Variable Projection Networks

import torch
from torch import nn
import torchvision.transforms.functional as TF
from utils.my_utils import plot_im_gray

class vp_layer2(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pad, y_old_shape, k_shape, noise_level, regularizer, x):
        # Wiener-deconvolution
        y_fft = torch.fft.fft2(y_pad)
        x = TF.center_crop(x, (y_fft.shape[-2], y_fft.shape[-1])) # pad to size
        #x = torch.fft.ifftshift(x, dim=(-2, -1))
        x_fft = torch.fft.fft2(x)
        x_fft_abs = torch.abs(x_fft)**2
        
        denom = noise_level * x_fft_abs + regularizer
        
        x_fft_inv = torch.conj(x_fft) / denom
        k_fft = noise_level * x_fft_inv * y_fft
        
        covariance_fft = 1 / denom
        covariance = torch.fft.ifft2(covariance_fft).real
        
        k = torch.fft.ifft2(k_fft).abs()
        #k = torch.fft.ifftshift(k, dim=(-2,-1))
        k = TF.center_crop(x, k_shape)

        y_estimate = torch.fft.ifft2(k_fft * x_fft).abs()
        y_estimate = TF.center_crop(y_estimate, y_old_shape)

        return k, covariance, y_estimate