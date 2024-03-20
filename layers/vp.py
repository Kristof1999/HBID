# based on VPNet: Variable Projection Networks

import torch
from torch import nn
import torchvision.transforms.functional as TF

class vp_layer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pad, y_old_shape, noise_level, regularizer, k):
        # Wiener-deconvolution
        y_fft = torch.fft.fft2(y_pad)
        k = TF.center_crop(k, (y_fft.shape[-2], y_fft.shape[-1])) # pad to size
        k = torch.fft.ifftshift(k, dim=(-2, -1))
        k_fft = torch.fft.fft2(k)
        k_fft_abs = torch.abs(k_fft)**2
        
        denom = noise_level * k_fft_abs + regularizer
        
        psf_fft_inv = torch.conj(k_fft) / denom
        x_fft = noise_level * psf_fft_inv * y_fft
        
        covariance_fft = 1 / denom
        covariance = torch.fft.ifft2(covariance_fft).real
        
        x = torch.fft.ifft2(x_fft).abs()
        x = TF.center_crop(x, y_old_shape)

        y_estimate = torch.fft.ifft2(k_fft * x_fft).abs()
        y_estimate = TF.center_crop(y_estimate, y_old_shape)

        return x, covariance, y_estimate