import torch
from torch import nn
from torch.autograd.function import Function
import torchvision.transforms.functional as TF
from torchvision.io import read_image, ImageReadMode

def my_edgetaper(x, pad, device):
    x = TF.pad(x, pad, padding_mode='symmetric')
    w = torch.hamming_window(pad*2).to(device)
    s = w.shape[0]
    w_lr = w[:s//2]
    #w_lr = torch.arange(0, 1, step=1/pad, device=device)
    x[:, :, :, :pad] = x[:, :, :, :pad] * w_lr
    x[:, :, :, -pad:] = x[:, :, :, -pad:] * w_lr.flip(dims=(0,))
    w_ud = w_lr[:, None]
    x[:, :, :pad, :] = x[:, :, :pad, :] * w_ud
    x[:, :, -pad:, :] = x[:, :, -pad:, :] * w_ud.flip(dims=(0,))
    return x

class vpfun_levin(Function):
    @staticmethod
    def forward(ctx, P, blurred_im, blurred_im_old_shape, isig_noise, reg):
        ctx.device = blurred_im.device
        ctx.s = blurred_im_old_shape
        ctx.s_p = (P.shape[-2], P.shape[-1])
        ctx.pad = P.shape[-1]//2
        ctx.isig_noise = isig_noise

        # Wiener-deconvolution
        #blurred_im = my_edgetaper(blurred_im[None, :, :, :], ctx.pad, ctx.device)
        Fblurred_im = torch.fft.fft2(blurred_im)
        P = TF.center_crop(P, (Fblurred_im.shape[-2], Fblurred_im.shape[-1])) # pad to size
        P = torch.fft.ifftshift(P, dim=(-2, -1))
        FP = torch.fft.fft2(P)
        abs_FP = torch.abs(FP)**2
        
        denom = isig_noise * abs_FP + reg
        Fpinv = FP.conj() / denom
        Fim_est = isig_noise * Fpinv * Fblurred_im

        C = 1 / denom
        C = torch.fft.ifft2(C).real
        
        im_est = torch.fft.ifft2(Fim_est).real
        im_est = TF.center_crop(im_est, ctx.s)

        blurred_im_est = torch.fft.ifft2(FP * Fim_est).real
        blurred_im_est = TF.center_crop(blurred_im_est, blurred_im_old_shape)

        ctx.save_for_backward(FP, Fblurred_im, denom)
        return im_est, C, blurred_im_est

    @staticmethod
    def backward(ctx, dJ_dout, dJ_dCov, dJ_dBlurred):
        # dJ_dout: derivative of loss function w.r.t. image estimate
        # dJ_dloss: derivative of loss function w.r.t. vp loss
        FP, Fblurred_im, denom = ctx.saved_tensors

        # problem: doesn't work with input size 100, but works with size 50
        # ideas:
        # - bigger size input contains tricky data, which cannot be found in the smaller size
        # - fft padding is worse for the bigger size?
        # - some computation performs with worse accuracy with bigger size
        # - bigger input is more noisy?

        # src: Learning to deblur appendix
        # TODO: try different ways to padding
        dJ_dout = TF.center_crop(dJ_dout, (Fblurred_im.shape[-2], Fblurred_im.shape[-1])) # pad to size
        dJ_dout = torch.fft.ifftshift(dJ_dout, dim=(-2,-1))
        FdJ_dout = torch.fft.fft2(dJ_dout)
        A = (Fblurred_im / denom)
        #B = (ctx.isig_noise * FP**2 * Fblurred_im.conj() / denom**2)
        B = (ctx.isig_noise * FP.conj()**2 * Fblurred_im / denom**2) # FP.conj() * dFP
        C = (ctx.isig_noise * FP.abs()**2  * Fblurred_im / denom**2) # dFP.conj() * FP
        #Fpinv_dP = Fblurred_im * (A - B - C)
        Fdout_dP = FdJ_dout.conj() * A - FdJ_dout * B.conj() - FdJ_dout.conj() * C
        Fdout_dP = ctx.isig_noise * Fdout_dP
        #Fdout_dP = (FdJ_dout * A.conj()).conj() - FdJ_dout * B.conj() - (FdJ_dout * C.conj()).conj()
        dJ_dP = torch.fft.ifft2(Fdout_dP).real
        #dJ_dP = torch.fft.fftshift(dJ_dP, dim=(-2,-1))
        
        dJ_dCov = TF.center_crop(dJ_dCov, (Fblurred_im.shape[-2], Fblurred_im.shape[-1])) # pad to size
        dJ_dCov = torch.fft.ifftshift(dJ_dCov, dim=(-2,-1))
        FdJ_dCov = torch.fft.fft2(dJ_dCov)
        B = (ctx.isig_noise * FP.conj() / denom**2) # FP.conj() * dFP
        C = (ctx.isig_noise * FP / denom**2) # dFP.conj() * FP
        FdCov_dP = - FdJ_dCov * B.conj() - FdJ_dCov.conj() * C
        dCov_dP = torch.fft.ifft2(FdCov_dP).real
        #dCov_dP = torch.fft.fftshift(dCov_dP, dim=(-2,-1))
        dJ_dP += dCov_dP
        
        dJ_dBlurred = TF.center_crop(dJ_dBlurred, (Fblurred_im.shape[-2], Fblurred_im.shape[-1])) # pad to size
        dJ_dBlurred = torch.fft.ifftshift(dJ_dBlurred, dim=(-2,-1))
        FdJ_dBlurred = torch.fft.fft2(dJ_dBlurred)
        A = FP.conj() * Fblurred_im / denom # FP.conj() * dFP
        B = FP * Fblurred_im / denom # dFP.conj() * FP
        C = (ctx.isig_noise * FP * FP.conj()**2 * Fblurred_im / denom**2) # FP.conj() * dFP
        D = (ctx.isig_noise * FP * FP.abs()**2  * Fblurred_im / denom**2) # dFP.conj() * FP
        FdBlurred_dP = FdJ_dBlurred * A.conj() + FdJ_dBlurred.conj() * B - FdJ_dBlurred * C.conj() - FdJ_dBlurred.conj() * D
        FdBlurred_dP = ctx.isig_noise * FdBlurred_dP
        #FdBlurred_dP = FdJ_dBlurred * Fim_est + FP * FdIm_dP
        dBlurred_dP = torch.fft.ifft2(FdBlurred_dP).real
        #dBlurred_dP = torch.fft.fftshift(dBlurred_dP, dim=(-2,-1))
        dJ_dP += dBlurred_dP
        
        #dJ_din = torch.fft.ifft2(FP * FdJ_dout.conj() / denom).real
        #dJ_din = torch.fft.fftshift(dJ_din, dim=(-2,-1))
        #dJ_din = ctx.isig_noise * dJ_din
        
        dJ_dP = TF.center_crop(dJ_dP, ctx.s_p)
        #dJ_din = TF.center_crop(dJ_din, ctx.s)

        return dJ_dP, None, None, None, None, None, None

path = "/home/kristof/Desktop/labor/ANBID/datasets/levin/"
y = read_image(path + f"im1_kernel1_img.png", ImageReadMode.GRAY)
y = (y/255).to(dtype=torch.double)

#psf = torch.rand((5,5), dtype=torch.double)
psf = torch.ones((5,5), dtype=torch.double)
psf = psf / torch.sum(psf)
psf = nn.Parameter(psf)

blurred_im = y[:, :100, :100]

blurred_im_old_shape = (blurred_im.shape[-2], blurred_im.shape[-1])
input = (psf, blurred_im, blurred_im_old_shape, 5, 10)
print(torch.autograd.gradcheck(vpfun_levin.apply, input))