import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
from PIL.Image import BILINEAR, BICUBIC
import os

real_in_dir = 'A:/Users/SSY/Documents/GitHub/PyTorch-CycleGAN/datasets/image_calibration/test/A'
real_out_dir = 'A:/Users/SSY/Documents/GitHub/PyTorch-CycleGAN/datasets/image_calibration/test/B'
fake_out_dir = 'A:/Users/SSY/Documents/GitHub/PyTorch-CycleGAN/output/B'

real_in_list = os.listdir(real_in_dir)
real_out_list = os.listdir(real_out_dir)
fake_out_list = os.listdir(fake_out_dir)

image_show_col = 50

plt_images = list(zip(real_in_list, real_out_list, fake_out_list))

plt.axis('off')
fig = plt.figure(figsize=(image_show_col*2/5, image_show_col*2))

for idx, image_zip in enumerate(plt_images[:image_show_col]):
    real_in = os.path.join(real_in_dir, image_zip[0])
    real_out = os.path.join(real_out_dir, image_zip[1])
    fake_out = os.path.join(fake_out_dir, image_zip[2])

    f_real_in = pil.open(real_in)
    f_real_out = pil.open(real_out)
    f_fake_out = pil.open(fake_out)

    w, h = f_real_out.size
    f_fake_out = f_fake_out.resize((w, h), BICUBIC)

    for i in range(2, -1, -1):
        ax = fig.add_subplot(image_show_col, 3, (idx+1)*3 - i)
        if i == 0:
            ax.imshow(f_fake_out)
        if i == 1:
            ax.imshow(f_real_out)
        if i == 2:
            ax.imshow(f_real_in)
        ax.axis('off')


fig.savefig('zzz.jpg', dpi=240)

