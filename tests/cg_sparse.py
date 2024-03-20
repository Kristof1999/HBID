import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def tikhonov(x):
    return torch.sum(x.abs()**2).item()

def make_BTTB(P, r, c, device=None):
    p_r = P.shape[-2]
    p_c = P.shape[-1]
    mid_row = p_r//2
    mid_col = p_c//2
    len = r*c

    x = torch.arange(end=r)
    y = torch.arange(end=c)
    [I, J] = torch.meshgrid(x, y, indexing='ij')

    row_vals = torch.arange(start=-mid_row, end=mid_row+1)
    col_vals = torch.arange(start=-mid_col, end=mid_col+1)
    row_idx = torch.zeros((r, c, p_r, p_c), dtype=torch.int)
    col_idx = torch.zeros((r, c, p_r, p_c), dtype=torch.int)
    for i in range(p_r):
        for j in range(p_c):
            row_idx[:, :, i, j] = I+row_vals[i]
            col_idx[:, :, i, j] = J+col_vals[j]

    row_idx = row_idx % r
    col_idx = col_idx % c
    row_mask = (0 <= row_idx) & (row_idx < r)
    col_mask = (0 <= col_idx) & (col_idx < c)
    mask = row_mask & col_mask
    P2 = P[None, None, :, :].expand((r, c, p_r, p_c))
    P2 = P2.to(device) # maybe not needed?

    idxs = r*I + J
    occurence = torch.sum(mask, (2, 3))
    i = torch.repeat_interleave(idxs.ravel(), occurence.ravel())
    #i = torch.repeat_interleave(idxs.ravel(), p_r*p_c)
    i = i.long()
    #j = r*row_idx[mask] + col_idx[mask]
    j = r*row_idx + col_idx
    j = j.flatten().long()
    v = P2[mask]
    BTTB = torch.sparse_coo_tensor(torch.stack((i, j)), v, (len, len),
                               dtype=torch.float, device=device)
    return BTTB

im = torch.zeros((150,150), dtype=torch.float)
im[50:100, 50] = 1
im[50:100, 100] = 1
im[50, 50:100] = 1
im[100, 50:100] = 1
im = im[None, :, :]

k = 1/9 * torch.ones((3,3), dtype=torch.float)

lambd = 1e-3
blurred_im = torch.conv2d(im, k[None, None, :, :], padding='same') + lambd * torch.randn((im.shape[-2], im.shape[-1]))

row = 150
col = 150
A = make_BTTB(k, row, col)
x = torch.zeros((row,col)).flatten()
AT = A.transpose(-2,-1)
Ax = AT @ (A @ x)
Ax += lambd*x

b = AT @ blurred_im.flatten()

r = b - Ax

# src: levin
# solve: Tk^T @ Tk @ x = Tk^T @ y, 
# where Tk is a Toeplitz matrix, k and y are given.
# A = Tk^T @ Tk, b = Tk^T @ y,
for i in range(50):
    rho = r.T @ r
    print(rho)
    if rho < 1e-5:
        print('break')
        break

    if i > 1:
        beta = rho / rho_1
        p = r + beta*p
    else:
        p = r
    
    Ap = AT @ (A @ p)
    Ap += lambd*p
    q = Ap
    alpha = rho / (p.T @ q)
    x = x + alpha*p
    r = r - alpha*q
    rho_1 = rho

mse = torch.nn.MSELoss()
x = x.reshape((row,col)).abs()
loss = mse(x, im).item()

plt.figure()
plt.imshow(im.squeeze(), cmap='gray')
plt.figure()
plt.imshow(blurred_im.squeeze(), cmap='gray')
plt.figure(f'MSE: {round(loss, 3)}')
plt.imshow(x.squeeze(), cmap='gray')
plt.show(block=True)