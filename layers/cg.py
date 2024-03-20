# based on VPNet: Variable Projection Networks

import torch
from torch import nn
import torchvision.transforms.functional as TF

class vp_layer_conjgrad(nn.Module):
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
        Fpinv = torch.conj(k_fft) / denom
        x_fft = y_fft * Fpinv

        # conjgrad
        lambd = 0.025
        Ax = x_fft * k_fft * k_fft.conj()
        Ax += lambd*x_fft
        b = y_fft * k_fft.conj()
        r = torch.fft.ifft2(b - Ax).abs()
        Ax = Ax.squeeze()
        b = b.squeeze()
        x = torch.fft.ifft2(x_fft).abs()

        # src: Levin
        # solve: Tk^T @ Tk @ x = Tk^T @ y, 
        # where Tk is a Toeplitz matrix, k and y are given.
        # A = Tk^T @ Tk, b = Tk^T @ y,
        for i in range(50):
            rho = torch.sum(r**2)
            if rho < 1e-15:
                break

            if i > 1:
                beta = rho / rho_1
                p = r + beta*p
            else:
                p = r
            
            Ap = torch.fft.ifft2(torch.fft.fft2(p) * k_fft * k_fft.conj()).real
            Ap += lambd*p
            q = Ap
            alpha = rho / torch.sum(p * q)
            x = x + alpha*p
            r = r - alpha*q
            rho_1 = rho
        
        return x