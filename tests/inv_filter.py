import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

im = torch.zeros((150,150), dtype=torch.float)
im[50:100, 50] = 1
im[50:100, 100] = 1
im[50, 50:100] = 1
im[100, 50:100] = 1

k = 1/9 * torch.ones((3,3), dtype=torch.float)

blurred_im = torch.conv2d(im[None, :, :], k[None, None, :, :])

blurred_im_fft = torch.fft.fft2(blurred_im)
k = TF.center_crop(k, (blurred_im.shape[-2], blurred_im.shape[-1]))
k = torch.fft.ifftshift(k)
k_fft = torch.fft.fft2(k)
im_est_fft = blurred_im_fft / k_fft
im_est = torch.fft.ifft2(im_est_fft).real
#im_est = torch.fft.ifftshift(im_est)

plt.figure()
plt.imshow(im, cmap='gray')
plt.figure()
plt.imshow(blurred_im.squeeze(), cmap='gray')
plt.figure()
plt.imshow(im_est.squeeze(), cmap='gray')
plt.show(block=True)