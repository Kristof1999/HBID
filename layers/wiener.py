import torch
from torch import nn
import torchvision.transforms.functional as TF

class wiener_layer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pad, y_old_shape, k):
        # Wiener-deconvolution
        y_fft = torch.fft.fft2(y_pad)
        k = TF.center_crop(k, (y_fft.shape[-2], y_fft.shape[-1])) # pad to size
        k = torch.fft.ifftshift(k, dim=(-2, -1))
        k_fft = torch.fft.fft2(k)
        k_fft_abs = torch.abs(k_fft)**2
        
        denom = k_fft_abs + 0.025
        k_fft_inv = torch.conj(k_fft) / denom
        x_fft = k_fft_inv * y_fft
        
        x = torch.fft.ifft2(x_fft).abs()
        x = TF.center_crop(x, y_old_shape)

        return x