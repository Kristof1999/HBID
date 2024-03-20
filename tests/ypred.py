import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

im = torch.zeros((150,150), dtype=torch.float)
im[50:100, 50] = 1
im[50:100, 100] = 1
im[50, 50:100] = 1
im[100, 50:100] = 1

#k = 1/9 * torch.ones((3,3), dtype=torch.float)
k = torch.ones((3,3), dtype=torch.float)
k[1, 1] = 3
k[0, 0] = 2
k[2, 2] = 2
k = k/k.sum()

lambd = 1e-3
n = lambd * torch.randn((im.shape[-2], im.shape[-1]))
blurred_im = torch.conv2d(im[None, None, :, :], k[None, None, :, :], padding='same') + n

blurred_im_fft = torch.fft.fft2(blurred_im)
k = TF.center_crop(k, (blurred_im.shape[-2], blurred_im.shape[-1]))
k = torch.fft.ifftshift(k)
k_fft = torch.fft.fft2(k)
im_est_fft = blurred_im_fft * k_fft.conj() / (k_fft.abs()**2 + lambd)
im_est = torch.fft.ifft2(im_est_fft).real
#im_est = torch.fft.ifftshift(im_est)

y_est_fft = k_fft * im_est_fft
y_est = torch.fft.ifft2(y_est_fft).real

y_est_fft_conj = k_fft.conj() * im_est_fft
y_est_conj = torch.fft.ifft2(y_est_fft_conj).real

plt.figure()
plt.imshow(im, cmap='gray')
plt.figure()
plt.imshow(blurred_im.squeeze(), cmap='gray')
plt.figure('Sharp im est')
plt.imshow(im_est.squeeze(), cmap='gray')
plt.figure('Blurred im est')
plt.imshow(y_est.squeeze(), cmap='gray')
plt.figure('Blurred im est conj')
plt.imshow(y_est_conj.squeeze(), cmap='gray')
plt.show(block=True)