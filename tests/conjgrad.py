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

Ax = im_est_fft * k_fft * k_fft.conj()
Ax += lambd*im_est_fft
b = blurred_im_fft * k_fft.conj()
r = torch.fft.ifft2(b - Ax).abs()
Ax = Ax.squeeze()
b = b.squeeze()
x = torch.fft.ifft2(im_est_fft.squeeze()).abs()

# src: levin
# solve: Tk^T @ Tk @ x = Tk^T @ y, 
# where Tk is a Toeplitz matrix, k and y are given.
# A = Tk^T @ Tk, b = Tk^T @ y,
for i in range(50):
    rho = torch.sum(r**2) #r.T @ r
    print(rho)
    if rho < 1e-15:
        print('break')
        break

    if i > 1:
        beta = rho / rho_1
        p = r + beta*p
    else:
        p = r
    
    Ap = torch.fft.ifft2(torch.fft.fft2(p) * k_fft * k_fft.conj()).real
    Ap += lambd*p
    #Ap = torch.conv2d(torch.conv2d(p, k.rot90(k=2, dims=(-2,-1)), padding='same'), k, padding='same')
    q = Ap
    alpha = rho / torch.sum(p * q) #(p.T @ q)
    x = x + alpha*p
    r = r - alpha*q
    rho_1 = rho

#x = torch.fft.ifft2(x).real
mse = torch.nn.MSELoss()
loss = mse(x, im).item()

plt.figure()
plt.imshow(im.squeeze(), cmap='gray')
plt.figure()
plt.imshow(blurred_im.squeeze(), cmap='gray')
plt.figure(f'MSE: {round(loss, 3)}')
plt.imshow(x.squeeze(), cmap='gray')
plt.show(block=True)