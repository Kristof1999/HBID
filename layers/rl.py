import torch
from torch import nn
import torchvision.transforms.functional as TF

class vp_layer_RL(nn.Module):
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
        x_fft = y_fft * k_fft_inv
        x = torch.fft.ifft2(x_fft).abs()

        k_fft = k_fft
        
        # Richardson-Lucy, src: wikipedia
        for i in range(50):
            denom_fft = torch.fft.fft2(x) * k_fft
            div = y_pad / torch.fft.ifft2(denom_fft).abs()
            mul_fft = torch.fft.fft2(div) * k_fft.conj()
            mul = torch.fft.ifft2(mul_fft).abs()
            x = x * mul

        x = TF.center_crop(x, y_old_shape)

        return x