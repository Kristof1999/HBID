# based on SelfDeblur, src: https://github.com/csdwren/SelfDeblur

import os
import torch
import torch.optim
import glob
from torchvision.io import write_png, read_image, ImageReadMode
import warnings
from utils.common_utils import *
from utils.my_utils import *
import time
from layers.vp import vp_layer
from layers.wiener import wiener_layer
from loss.loss_levin_ypred import LevinLossYPred
import logging
from networks.fcn import fcn

def run_mlp():
    var_name = 'mlp'
    local_time = time.localtime()
    logging.basicConfig(filename=f"results/levin/{var_name}/{var_name}_{local_time.tm_year}:{local_time.tm_mon}:{local_time.tm_mday}:{local_time.tm_hour}:{local_time.tm_min}.txt", level=logging.INFO)
    opt = parse_helper(var_name)
    print_freq = opt.print_frequency
    LR = opt.lr
    LR = 1e-3
    num_iter = opt.num_iter
    logging.info(opt)

    os.environ["HSA_OVERRIDE_GFX_VERSION"]="10.3.0"
    device = 'cuda'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    warnings.filterwarnings("ignore")

    files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
    files_source.sort()
    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)

    times = []

    # start #image
    for f in files_source:
        imgname = os.path.basename(f)
        imgname = os.path.splitext(imgname)[0]

        opt.kernel_size, kernel_name = get_kernel(imgname)
        y = read_image(opt.data_path + f"/{imgname}.png", ImageReadMode.GRAY)
        y = normalize(y).to(device, dtype=torch.float)
        y = y[None, :, :, :]

        print(imgname)
        logging.info(imgname)

        vp = vp_layer()
        loss_fn = LevinLossYPred()
        
        INPUT = 'noise'
        pad = 'reflection'
        dtype = torch.cuda.FloatTensor
        n_k = 200
        net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
        net_input_kernel.squeeze_()

        net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
        net_kernel = net_kernel.type(dtype)
        
        optimizer = torch.optim.Adam([{'params':net_kernel.parameters()}], lr=LR)

        pad = opt.kernel_size[-1]//2
        y_old_shape = (y.shape[-2], y.shape[-1])
        y_pad = my_edgetaper(y, pad, device)
        reg_sum = reg_sum_helper(y_pad.shape, device=device)

        high = 0.1
        low = 0.001
        noise_levels = torch.range(high, low, -(high-low)/num_iter)
        high = 100
        low = 30
        prior_ivars = torch.range(high, low, -(high-low)/num_iter)
        
        start = time.time()
        for step in range(num_iter):
            optimizer.zero_grad()

            sig_noise = noise_levels[step]
            noise_level = 1/sig_noise**2
            ivar = prior_ivars[step]
            regularizer = ivar * reg_sum

            k = net_kernel(net_input_kernel)
            k = torch.reshape(k, opt.kernel_size)
            x, covariance, y_estimate = vp(y_pad, y_old_shape, noise_level, regularizer, k)
            total_loss = loss_fn(y_estimate, y, covariance, k)
            
            total_loss.backward()
            optimizer.step()

            if (step+1) % print_freq == 0:
                print(step+1, total_loss.item())
                logging.info(f"loss: {step+1} - {total_loss.item()}")
                #save_helper(k, f"{local_time.tm_hour}:{local_time.tm_min}_{imgname}_{step}_k.png", save_path)

        end = time.time()
        res = end-start
        times.append(res)
        logging.info(f"time: {imgname} - {res}")

        wiener = wiener_layer()
        out_x = wiener(y_pad, y_old_shape, k)
        
        save_helper(k, f"{imgname}_k.png", save_path)
        save_helper(out_x, f"{imgname}_x.png", save_path)

    times = torch.tensor(times)
    print("AVG:", times.mean())