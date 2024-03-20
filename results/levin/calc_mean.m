name = "ypred";
maxshift = 5;

mean_im_one_kernel = zeros(1, 8);
mean_kernel_all_im = zeros(1, 8);
for k = 1:8
    mean_kssim = zeros(1,4);
    mean_xssim = zeros(1,4);
    for x = 1:4
        tk = imread(name+"/im"+x+"_kernel"+k+"_img_k.png");
        k2 = imread("gt/kernel"+k+".png");
        [r1, c1] = size(tk);
        [r2, c2] = size(k2);
        %tk = padarray(tk, [floor((r2-r1)/2), floor((c2-c1)/2)]);
        %tk = im2double(tk);
        %k2 = im2double(k2);
        %[p, s, e] = comp_upto_shift(tk, k2, maxshift);
        %mean_kssim(x) = s;%ssim(tk, k2);

        tx = imread(name+"/im"+x+"_kernel"+k+"_img_x_cg.png");
        x2 = imread("gt/im"+x+".png");
        [r2, c2] = size(tx);
        [r1, c1] = size(x2);
        tx = tx(floor((r2-r1)/2)+1:end-floor((r2-r1)/2), floor((c2-c1)/2)+1:end-floor((c2-c1)/2));
        tx = im2double(tx);
        x2 = im2double(x2);
        [p, s, e] = comp_upto_shift(tx, x2, maxshift);
        mean_xssim(x) = s;%ssim(tx, x2);
        %figure, imshow([tx, x2])
    end
    mean_kernel_all_im(k) = mean(mean_kssim);
    mean_im_one_kernel(k) = mean(mean_xssim);
end
mean(mean_im_one_kernel)
mean(mean_kernel_all_im)

mean_im_all_kernel = zeros(1,4);
mean_kernel_one_im = zeros(1,4);
for x = 1:4
    mean_xssim = zeros(1,8);
    mean_kssim = zeros(1,8);
    for k = 1:8
        tk = imread(name+"/im"+x+"_kernel"+k+"_img_k.png");
        k2 = imread("gt/kernel"+k+".png");
        [r1, c1] = size(tk);
        [r2, c2] = size(k2);
        tk = padarray(tk, [floor((r2-r1)/2), floor((c2-c1)/2)]);
        %mean_kssim(k) = ssim(tk, k2);

        tx = imread(name+"/im"+x+"_kernel"+k+"_img_x_cg.png");
        x2 = imread("gt/im"+x+".png");
        [r2, c2] = size(tx);
        [r1, c1] = size(x2);
        tx = tx(floor((r2-r1)/2)+1:end-floor((r2-r1)/2), floor((c2-c1)/2)+1:end-floor((c2-c1)/2));
        tx = im2double(tx);
        x2 = im2double(x2);
        [p, s, e] = comp_upto_shift(tx, x2, maxshift);
        mean_xssim(k) = s;%ssim(tx, x2);
        %mean_xssim(k) = ssim(tx, x2);
    end
    mean_kernel_one_im(x) = mean(mean_kssim);
    mean_im_all_kernel(x) = mean(mean_xssim);
end
mean(mean_im_all_kernel)
mean(mean_kernel_one_im)

% how good is image estimate for a given kernel?
figure
bar(mean_im_one_kernel)
title("Image given kernel")

% how good is image estimate for all kernels?
% figure
% bar(mean_im_all_kernel)
% title("Image for all kernels")
% 
% % how good is kernel estimate for a given image?
% figure
% bar(mean_kernel_one_im)
% title("Kernel given image")
% 
% % how good is kernel estimate for all images?
% figure
% bar(mean_kernel_all_im)
% title("Kernel for all image")