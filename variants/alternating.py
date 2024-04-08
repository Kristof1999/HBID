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
from layers.vp2 import vp_layer2
from layers.wiener import wiener_layer
from loss.loss_levin_ypred import LevinLossYPred
from loss.loss_levin_x import LevinLossX
import logging

def run_alternating():
    var_name = 'alternating'
    local_time = time.localtime()
    logging.basicConfig(filename=f"results/levin/{var_name}/{var_name}_{local_time.tm_year}:{local_time.tm_mon}:{local_time.tm_mday}:{local_time.tm_hour}:{local_time.tm_min}.txt", level=logging.INFO)
    opt = parse_helper(var_name)
    print_freq = opt.print_frequency
    LR = opt.lr
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
        vp2 = vp_layer2()
        mse = nn.MSELoss()
        loss_fn = LevinLossYPred()
        loss_fn2 = LevinLossX()
        torch.manual_seed(0) # important!
        k = torch.randn(opt.kernel_size, device=device, dtype=torch.float)
        k[opt.kernel_size[-2]//2, opt.kernel_size[-1]//2] = 0.5
        k[opt.kernel_size[-2]//2, opt.kernel_size[-1]//2+1] = 0.5
        x = torch.tensor(y)

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
            sig_noise = noise_levels[step]
            noise_level = 1/sig_noise**2
            ivar = prior_ivars[step]
            regularizer = ivar * reg_sum
            
            # x opt, k fix
            k = torch.softmax(k.flatten(), dim=-1).reshape(opt.kernel_size)
            x, covariance, y_estimate = vp(y_pad, y_old_shape, noise_level, regularizer, k)
            total_loss = loss_fn(y_estimate, y, covariance, k)
            
            # x fix, k opt
            k, covariance, y_estimate = vp2(y_pad, y_old_shape, opt.kernel_size, noise_level, regularizer, x)
            total_loss += mse(y_estimate, y)

            if (step+1) % print_freq == 0:
                plot_im_gray(k)
                plot_im_gray(x)
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
        save_helper(out_x, f"{imgname}_x_wiener.png", save_path)
        save_helper(x, f"{imgname}_x.png", save_path)

    times = torch.tensor(times)
    print("AVG:", times.mean())