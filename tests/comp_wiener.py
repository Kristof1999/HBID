import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

im = torch.zeros((150,150), dtype=torch.float)
im[50:100, 50] = 1
im[50:100, 100] = 1
im[50, 50:100] = 1
im[100, 50:100] = 1
im = im[None, :, :]

k = 1/9 * torch.ones((3,3), dtype=torch.float)

lambd = 1e-3
blurred_im = torch.conv2d(im, k[None, None, :, :], padding='same') + lambd * torch.randn((im.shape[-2], im.shape[-1]))

blurred_im_fft = torch.fft.fft2(blurred_im)
k = TF.center_crop(k, (blurred_im.shape[-2], blurred_im.shape[-1]))
k = torch.fft.ifftshift(k)
k_fft = torch.fft.fft2(k)

im_est_fft_big_reg = blurred_im_fft * k_fft.conj() / (k_fft.abs()**2 + lambd*32)
im_est_big_reg = torch.fft.ifft2(im_est_fft_big_reg).abs()

im_est_fft_small_reg = im_est_fft_big_reg * k_fft.conj() / (k_fft.abs()**2 + lambd/32)
im_est_small_reg = torch.fft.ifft2(im_est_fft_small_reg).abs()

plt.figure()
plt.imshow(im.squeeze(), cmap='gray')
plt.figure()
plt.imshow(blurred_im.squeeze(), cmap='gray')
plt.figure()
plt.imshow(im_est_big_reg.squeeze(), cmap='gray')
plt.figure()
plt.imshow(im_est_small_reg.squeeze(), cmap='gray')
plt.show(block=True)