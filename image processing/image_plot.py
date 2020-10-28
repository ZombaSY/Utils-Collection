import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import os
import math

from PIL.Image import BILINEAR, BICUBIC

"""
:Usage
 1. set 'real_in_dir', 'real_out_dir', 'fake_out_dir'
 2. set 'image_show_row' to limit the maximum row in plots
 3. if "output image" needs resize the same ratio of "input or target image", release annotation near line 62
"""

plt.axis('off')
figure_dpi = 300

# limit the maximum row to avoid 'pyplot plot maximum size exception'.
PLT_ROW = 50
PLT_COL = 3

input_dir = 'A:/Users/SSY/Desktop/dataset/cud_calibration/201012 dataset/train/A'
output_dir = 'A:/Users/SSY/Desktop/temp'
target_dir = 'A:/Users/SSY/Desktop/dataset/cud_calibration/201012 dataset/train/B'

input_list = os.listdir(input_dir)
output_list = os.listdir(output_dir)
target_list = os.listdir(target_dir)

assert len(input_list) == len(output_list) == len(target_list), 'length of images in directory should be same'

image_show_row = len(input_list)
# image_show_row = 10

plt_images = list(zip(input_list, output_list, target_list))

render_iteration = 1
if image_show_row > PLT_ROW:
    render_iteration = math.ceil(image_show_row / PLT_ROW)

for render_idx in range(render_iteration):
    last_index = render_idx * PLT_ROW

    figure_width = 1280 / figure_dpi / 2
    figure_height = 1080 / figure_dpi

    if not (render_idx + 1 == render_iteration):
        plt_render_images = plt_images[last_index:PLT_ROW * (render_idx + 1)]

        fig = plt.figure(figsize=(figure_width * PLT_COL, figure_height * PLT_ROW))
        fig.dpi = figure_dpi
    else:
        plt_render_images = plt_images[last_index:image_show_row]

        _, lefts = divmod(image_show_row, PLT_ROW)
        fig = plt.figure(figsize=(figure_width * PLT_COL, figure_height * lefts))
        fig.dpi = figure_dpi

    for idx, image_zip in enumerate(plt_render_images):
        input_img_dr = os.path.join(input_dir, image_zip[0])
        output_img_dr = os.path.join(output_dir, image_zip[1])
        target_img_dr = os.path.join(target_dir, image_zip[2])

        input_img = pil.open(input_img_dr)
        output_img = pil.open(output_img_dr)
        target_img = pil.open(target_img_dr)

        w, h = input_img.size
        # output_img = input_img.resize((w, h), BICUBIC)   # if needs resize

        for i in range(2, -1, -1):
            ax = fig.add_subplot(image_show_row, PLT_COL, (idx+1) * PLT_COL - i)
            if i == 2:
                ax.imshow(input_img)
            if i == 1:
                ax.imshow(output_img)
            if i == 0:
                ax.imshow(target_img)
            ax.axis('off')

        print('finished plotting {}th image'.format((render_idx * PLT_ROW) + (idx + 1)))

    print('\nmaking image... wait for seconds')

    if not os.path.exists('./image_plot'):
        os.mkdir('./image_plot')

    fn = './image_plot/output_' + str(render_idx).zfill(3) + '.jpg'
    fig.savefig(fn, dpi=fig.dpi)
    print('saved', fn, end='\n\n')

print('Done!!!')
