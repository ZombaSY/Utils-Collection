from PIL import Image

import colorsys
import os


def m_rgb_to_hsv(src):
    if isinstance(src, Image.Image):
        r, g, b = src.split()

        h_dat = []
        s_dat = []
        v_dat = []

        for rd, gn, bl in zip(r.getdata(), g.getdata(), b.getdata()):
            h, s, v = colorsys.rgb_to_hsv(rd / 255., gn / 255., bl / 255.)
            h_dat.append(int(h * 255.))
            s_dat.append(int(s * 255.))
            v_dat.append(int(v * 255.))
        r.putdata(h_dat)
        g.putdata(s_dat)
        b.putdata(v_dat)

        return Image.merge('RGB', (r, g, b))
    else:
        return None


def m_hsv_to_rgb(src):
    if isinstance(src, Image.Image):
        r, g, b = src.split()

        h_dat = []
        s_dat = []
        v_dat = []

        for rd, gn, bl in zip(r.getdata(), g.getdata(), b.getdata()):
            h, s, v = colorsys.hsv_to_rgb(rd/255., gn/255., bl/255.)
            h_dat.append(int(h*255.))
            s_dat.append(int(s*255.))
            v_dat.append(int(v*255.))
        r.putdata(h_dat)
        g.putdata(s_dat)
        b.putdata(v_dat)

        return Image.merge('RGB', (r, g, b))
    else:
        return None


if __name__ == '__main__':
    dataset_path = 'dataset/'
    saved_path = 'hsv_rgb_conversion/'

    file_list = os.listdir(dataset_path)

    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    for idx, fn in enumerate(file_list):
        img_path = dataset_path + fn

        fn, _ = os.path.splitext(fn)

        img_origin = Image.open(img_path)
        img_hsv = m_rgb_to_hsv(img_origin)
        img_hsv.save(saved_path + fn + '_hsv.jpg')

        img_rgb = m_hsv_to_rgb(img_hsv)
        img_rgb.save(saved_path + fn + '_rgb.jpg')

        print(saved_path + fn + '\t  saved!')
