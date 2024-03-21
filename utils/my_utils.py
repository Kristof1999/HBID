import torch
import os
from torchvision.io import write_png
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import argparse

def parse_helper(name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
    parser.add_argument('--img_size', type=int, default=[255, 255], help='size of each image dimension')
    parser.add_argument('--kernel_size', type=int, default=[21, 21], help='size of blur kernel [height, width]')
    parser.add_argument('--data_path', type=str, default="datasets/levin/", help='path to blurry image')
    parser.add_argument('--save_path', type=str, default=f"results/levin/{name}", help='path to save results')
    parser.add_argument('--save_frequency', type=int, default=1000, help='frequency to save results')
    parser.add_argument('--print_frequency', type=int, default=500, help='frequency to print results')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=0.5)
    opt = parser.parse_args()
    print("Arguments:", opt)
    return opt

def reg_sum_helper(shape, device):
    dx = torch.tensor([[-1, 1], [0, 0]], device=device)
    dy = torch.tensor([[-1, 0], [1, 0]], device=device)
    dx = TF.center_crop(dx, (shape[-2], shape[-1]))
    dy = TF.center_crop(dy, (shape[-2], shape[-1]))
    dx_fft = torch.fft.fft2(dx)
    dy_fft = torch.fft.fft2(dy)
    dx_fft[0, 0] = max(dx_fft[0, 0].abs(), 0.001)
    dy_fft[0, 0] = max(dy_fft[0, 0].abs(), 0.001)
    dx_fft = dx_fft.abs()**2
    dy_fft = dy_fft.abs()**2
    return (dx_fft + dy_fft)/2

def reg_loss(out_x, device):
    #return l1l2(out_x)
    #return l1l2_der(out_x, device)
    return tikhonov(out_x)
    #return sobel(out_x, device)
    #return der(out_x, device)

def l1l2(out_x):
    return torch.mean(out_x.abs()) / torch.sqrt(torch.mean(out_x.abs()**2))

def l1l2_der(out_x, device):
    out_x = TF.gaussian_blur(out_x, (3,3), sigma=1)
    dx = torch.tensor([[-1, 1], [0, 0]], device=device, dtype=torch.float)
    dy = torch.tensor([[-1, 0], [1, 0]], device=device, dtype=torch.float)
    dx = dx[None, None, :, :]
    dy = dy[None, None, :, :]
    x_dx = torch.conv2d(out_x, dx).abs()
    x_dy = torch.conv2d(out_x, dy).abs()
    return x_dx.mean() / torch.mean(x_dx.abs()**2).sqrt() + x_dy.mean() / torch.mean(x_dy**2).sqrt()

def tikhonov(out_x):
    return torch.mean(out_x.abs()**2)

def der(out_x, device):
    out_x = TF.gaussian_blur(out_x, (3,3), sigma=1)
    dx = torch.tensor([[-1, 1], [0, 0]], device=device, dtype=torch.float)
    dy = torch.tensor([[-1, 0], [1, 0]], device=device, dtype=torch.float)
    dx = dx[None, None, :, :]
    dy = dy[None, None, :, :]
    x_dx = torch.conv2d(out_x, dx).abs()
    x_dy = torch.conv2d(out_x, dy).abs()
    return (torch.mean(x_dx**2) + torch.mean(x_dy**2))

def sobel(out_x, device):
    out_x = TF.gaussian_blur(out_x, (3,3), sigma=1)
    dx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=torch.float)
    dy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=torch.float)
    dx = dx[None, None, :, :]
    dy = dy[None, None, :, :]
    x_dx = torch.conv2d(out_x, dx).abs()
    x_dy = torch.conv2d(out_x, dy).abs()
    return (torch.mean(x_dx**2) + torch.mean(x_dy**2))

def plot_im_gray(x):
    plt.figure()
    plt.imshow(x.detach().cpu().squeeze(), cmap='gray')
    plt.show(block=True)

def normalize(x):
    return x/255

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

def save_helper(x, name, save_path):
    save_path = os.path.join(save_path, name)
    x = ((x-x.min()) / (x.max() - x.min())) * 255
    x = x.squeeze()
    x = x[None, :, :]
    write_png(x.cpu().byte(), save_path, 0)

def get_kernel(imgname):
    kernel_name = None
    kernel_size = None
    if imgname.find('kernel1') != -1:
        kernel_size = [17, 17]
        kernel_name = 'kernel1'
    if imgname.find('kernel2') != -1:
        kernel_size = [15, 15]
        kernel_name = 'kernel2'
    if imgname.find('kernel3') != -1:
        kernel_size = [13, 13]
        kernel_name = 'kernel3'
    if imgname.find('kernel4') != -1:
        kernel_size = [27, 27]
        kernel_name = 'kernel4'
    if imgname.find('kernel5') != -1:
        kernel_size = [11, 11]
        kernel_name = 'kernel5'
    if imgname.find('kernel6') != -1:
        kernel_size = [19, 19]
        kernel_name = 'kernel6'
    if imgname.find('kernel7') != -1:
        kernel_size = [21, 21]
        kernel_name = 'kernel7'
    if imgname.find('kernel8') != -1:
        kernel_size = [21, 21]
        kernel_name = 'kernel8'
    return kernel_size, kernel_name