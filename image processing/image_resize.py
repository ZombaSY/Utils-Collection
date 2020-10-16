from PIL import Image

import math
import os

DATASET_PATH = 'A:/Users/SSY/Desktop/dataset/cud_calibration/201012 dataset/train/A/'
MAXIMUM_RESOLUTION = 1280*720


def img_resize(img, maximum_resolution):
    img_width = img.width
    img_height = img.height
    img_definition = img_width * img_height
    img_dpi = img.info['dpi']

    if img_definition > maximum_resolution:
        reduction_ratio = img_definition / maximum_resolution

        reduction_ratio = math.sqrt(reduction_ratio)

        img_width_r = int(img_width / reduction_ratio)
        img_height_r = int(img_height / reduction_ratio)

        img = img.resize((img_width_r, img_height_r))

    return img, img_dpi


def main():
    file_list = os.listdir(DATASET_PATH)

    if not os.path.exists('./image_resize'):
        os.mkdir('.image_resize')

    for idx, fn in enumerate(file_list):
        img_path = DATASET_PATH + fn
        img = Image.open(img_path)
        img, dpi = img_resize(img, maximum_resolution=MAXIMUM_RESOLUTION)

        img.save('./image_resize/' + fn, quality=100, dpi=dpi)
        print(fn + ' Done!')


if __name__ == '__main__':
    main()
