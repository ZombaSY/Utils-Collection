import PIL.Image as Image
import os
import numpy as np

from matplotlib import pyplot as plt
from skimage import color


input_dir = 'dataset/T3-240_a.jpg'
output_dir = 'dataset/T3-240_out.jpg'
target_dir = 'dataset/T3-240_b.jpg'


def show_histogram(_input, _output, _target, title, save_image=False, bins=100):
    """
    :param _input: Numpy Array
    :param _output: Numpy Array
    :param _target: Numpy Array
    :param title: This variable decides the loop iterations. Choice of ['RGB', 'HSV', 'Lab'].
    :param save_image: Boolean
    :param bins: The number of 'x axis' in plot
    """

    for i in range(len(title)):

        x_hist_dim = np.histogram(_input[:, :, i], bins=bins)
        output_hist_dim = np.histogram(_output[:, :, i], bins=bins)
        target_hist_dim = np.histogram(_target[:, :, i], bins=bins)

        plt.plot(x_hist_dim[0], label=title[i] + '_input')
        plt.plot(output_hist_dim[0], label=title[i] + '_output')
        plt.plot(target_hist_dim[0], label=title[i] + '_target')

        plt.legend()
        if save_image:
            if not os.path.exists('show_histogram'):
                os.mkdir('show_histogram')
            plt.savefig('show_histogram/' + title + '_' + title[i])
        plt.show()
        plt.clf()


def main():

    x_rgb = np.array(Image.open(input_dir).convert('RGB'))
    output_rgb = np.array(Image.open(output_dir).convert('RGB'))
    target_rgb = np.array(Image.open(target_dir).convert('RGB'))

    x_hsv = color.rgb2hsv(x_rgb)
    output_hsv = color.rgb2hsv(output_rgb)
    target_hsv = color.rgb2hsv(target_rgb)

    x_lab = color.rgb2lab(x_rgb)
    output_lab = color.rgb2lab(output_rgb)
    target_lab = color.rgb2lab(target_rgb)

    show_histogram(x_rgb, output_rgb, target_rgb, title='RGB', save_image=True)
    show_histogram(x_hsv, output_hsv, target_hsv, title='HSV', save_image=True)
    show_histogram(x_lab, output_lab, target_lab, title='Lab', save_image=True)


if __name__ == '__main__':
    main()
