import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from SSIM import SSIM

def tikhonov(x):
    return torch.sum(x.abs()**2).item()

def derivative(x, alpha):
    out_x = TF.gaussian_blur(x, (3,3), sigma=1)
    dx = torch.tensor([[-1, 1], [0, 0]], dtype=torch.float)
    dy = torch.tensor([[-1, 0], [1, 0]], dtype=torch.float)
    dx = dx[None, None, :, :]
    dy = dy[None, None, :, :]
    x_dx = torch.conv2d(out_x, dx).abs()
    x_dy = torch.conv2d(out_x, dy).abs()
    return torch.sum(x_dx**alpha).item() + torch.sum(x_dy**alpha).item()

def my_edgetaper(x, pad, device):
    x = TF.pad(x, pad, padding_mode='symmetric')
    w = torch.hamming_window(pad*2).to(device)
    s = w.shape[0]
    w_lr = w[:s//2]
    x[:, :, :, :pad] = x[:, :, :, :pad] * w_lr
    x[:, :, :, -pad:] = x[:, :, :, -pad:] * w_lr.flip(dims=(0,))
    w_ud = w_lr[:, None]
    x[:, :, :pad, :] = x[:, :, :pad, :] * w_ud
    x[:, :, -pad:, :] = x[:, :, -pad:, :] * w_ud.flip(dims=(0,))
    return x

im = torch.zeros((150,150), dtype=torch.float)
im[50:100, 50] = 1
im[50:100, 100] = 1
im[50, 50:100] = 1
im[100, 50:100] = 1
im = im[None, None, :, :]

k = 1/9 * torch.ones((3,3), dtype=torch.float)

lambd = 1e-3
n = lambd * torch.randn((im.shape[-2], im.shape[-1]))
blurred_im = torch.conv2d(im, k[None, None, :, :], padding='same') + n

mse = torch.nn.MSELoss()
ssim = SSIM()

n = n[None, None, :, :]
pad = 3//2
old_shape = blurred_im.shape
blurred_im = my_edgetaper(blurred_im, pad, None)
blurred_im_fft = torch.fft.fft2(blurred_im)
k = TF.center_crop(k, (blurred_im.shape[-2], blurred_im.shape[-1]))
k = torch.fft.ifftshift(k)
k_fft = torch.fft.fft2(k)

def get_im_est(lambd):
    im_est_fft = blurred_im_fft * k_fft.conj() / (k_fft.abs()**2 + lambd)
    return TF.center_crop(torch.fft.ifft2(im_est_fft).abs(), (old_shape[-2], old_shape[-1]))

im2 = my_edgetaper(im, pad, None)
im_fft = torch.fft.fft2(im2)
n = my_edgetaper(n, pad, None)
n_fft = torch.fft.fft2(n)
im_est = get_im_est(n_fft.abs()**2 / im_fft.abs()**2)
mse1 = mse(im_est, im).item()
ssim1 = ssim(im_est, im).item()

im_est_constant = get_im_est(lambd)
mse_constant = mse(im_est_constant, im).item()
ssim_constant = ssim(im_est_constant, im).item()

im_est_big_reg = get_im_est(lambd*32)
mse_big = mse(im_est_big_reg, im).item()
ssim_big = ssim(im_est_big_reg, im).item()

im_est_small_reg = get_im_est(lambd/32)
mse_small = mse(im_est_small_reg, im).item()
ssim_small = ssim(im_est_small_reg, im).item()

im_est_squared = get_im_est(lambd**2)
mse_squared = mse(im_est_squared, im).item()
ssim_squared = ssim(im_est_squared, im).item()

print('----------Original:', 
      f'MSE: {round(mse1, 3)}', 
      f'Tikhonov: {round(tikhonov(im_est), 3)}', 
      f'Gauss derivative: {round(derivative(im_est, 2), 3)}', 
      f'0.5 derivative: {round(derivative(im_est, 0.5), 3)}', 
      f'SSIM: {round(ssim1, 5)}', 
      sep='\n')
print('----------Constant:', 
      f'MSE: {round(mse_constant, 3)}', 
      f'Tikhonov: {round(tikhonov(im_est_constant), 3)}', 
      f'Gauss derivative: {round(derivative(im_est_constant, 2), 3)}', 
      f'0.5 derivative: {round(derivative(im_est_constant, 0.5), 3)}', 
      f'SSIM: {round(ssim_constant, 5)}', 
      sep='\n')
print('----------Squared:', 
      f'MSE: {round(mse_squared, 3)}', 
      f'Tikhonov: {round(tikhonov(im_est_squared), 3)}', 
      f'Gauss derivative: {round(derivative(im_est_squared, 2), 3)}', 
      f'0.5 derivative: {round(derivative(im_est_squared, 0.5), 3)}', 
      f'SSIM: {round(ssim_squared, 5)}', 
      sep='\n')
print('----------Big regularization:', 
      f'MSE: {round(mse_big, 3)}', 
      f'Tikhonov: {round(tikhonov(im_est_big_reg), 3)}', 
      f'Gauss derivative: {round(derivative(im_est_big_reg, 2), 3)}', 
      f'0.5 derivative: {round(derivative(im_est_big_reg, 0.5), 3)}', 
      f'SSIM: {round(ssim_big, 5)}', 
      sep='\n')
print('----------Small regularization:', 
      f'MSE: {round(mse_small, 3)}', 
      f'Tikhonov: {round(tikhonov(im_est_small_reg), 3)}', 
      f'Gauss derivative: {round(derivative(im_est_small_reg, 2), 3)}', 
      f'0.5 derivative: {round(derivative(im_est_small_reg, 0.5), 3)}', 
      f'SSIM: {round(ssim_small, 5)}', 
      sep='\n')

plt.figure('Noise')
plt.imshow(n.squeeze(), cmap='gray')
plt.figure('Sharp')
plt.imshow(im.squeeze(), cmap='gray')
plt.figure('Blurred and noisy')
plt.imshow(blurred_im.squeeze(), cmap='gray')
plt.figure('Original')
plt.imshow(im_est.squeeze(), cmap='gray')
plt.figure('Constant')
plt.imshow(im_est_constant.squeeze(), cmap='gray')
plt.figure('Squared')
plt.imshow(im_est_squared.squeeze(), cmap='gray')
plt.figure('Big regulurization')
plt.imshow(im_est_big_reg.squeeze(), cmap='gray')
plt.figure('Small regularization')
plt.imshow(im_est_small_reg.squeeze(), cmap='gray')
plt.show(block=True)