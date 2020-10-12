from PIL import Image

import math
import os

DATASET_PATH = 'A:/Users/SSY/Desktop/dataset/cud_calibration/RAW/학습데이터(T3_81-131)_201008 전달/A/'
MAXIMUM_RESOLUTION = 1280*720


def img_resize(src_path, fn, maximum_resolution):
    img: Image.Image = Image.open(src_path)

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

    if not os.path.exists('./image_resize'):
        os.mkdir('.image_resize')

    img.save('./image_resize/' + fn, quality=100, dpi=img_dpi)
    print(fn + ' Done!')


def main():
    file_list = os.listdir(DATASET_PATH)

    for idx, fn in enumerate(file_list):
        img_path = DATASET_PATH + fn
        img_resize(img_path, fn, maximum_resolution=MAXIMUM_RESOLUTION)


if __name__ == '__main__':
    main()
