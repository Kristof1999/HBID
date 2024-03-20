import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def tikhonov(x):
    return torch.sum(x.abs()**2).item()

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
im_est_fft = blurred_im_fft * k_fft.conj() / (k_fft.abs()**2 + lambd)
im_est = torch.fft.ifft2(im_est_fft).abs()

x = torch.fft.ifft2(im_est_fft.squeeze()).abs()
x = x[None, None, :, :]
k = 1/9 * torch.ones((3,3), dtype=torch.float)
k = k[None, None, :, :]

plt.figure("Initial")
plt.imshow(x.squeeze(), cmap='gray')

# src: wikipedia
for i in range(10):
    #denom = torch.conv2d(x, k, padding='same')
    #div = blurred_im / denom
    #mul = torch.conv2d(div, k.rot90(k=2, dims=(-2,-1)), padding='same')
    denom_fft = torch.fft.fft2(x) * k_fft
    div = blurred_im / torch.fft.ifft2(denom_fft).abs()
    mul_fft = torch.fft.fft2(div) * k_fft.conj()
    mul = torch.fft.ifft2(mul_fft).abs()
    x = x * mul

mse = torch.nn.MSELoss()
loss = mse(x, im).item()

plt.figure("Original")
plt.imshow(im.squeeze(), cmap='gray')
plt.figure("Degraded")
plt.imshow(blurred_im.squeeze(), cmap='gray')
plt.figure(f'MSE: {round(loss, 3)}')
plt.imshow(x.squeeze(), cmap='gray')
plt.show(block=True)