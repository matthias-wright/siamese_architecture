
"""
This file contains auxiliary functions and classes that do not fit anywhere else.
"""

import matplotlib.pyplot as plt
from matplotlib import pylab
import sys
import numpy as np
from PIL import Image
import torch

plt.style.use('ggplot')


def plot_training_losses(d_losses, g_losses):
    """
    Creates a plot for the training losses of the discriminator and the generator
    and saves the figure as a PNG file.
    :param d_losses: (list) the losses of the discriminator.
    :param g_losses: (list) the losses of the generator.
    """
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.title('Wasserstein GAN: training losses')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig('training_losses.png')
    plt.clf()


def image_to_tensor(path):
    """
    Reads an image from disk space and converts it to a PyTorch tensor.
    :param path: (string) the file path specifying the location of the image.
    :return: (tensor) PyTorch tensor representing the image (normalized to [0, 1]).
    """
    img = np.asarray(Image.open(path))
    im_norm = img / 255
    im_norm = np.moveaxis(im_norm[np.newaxis], 3, 1)
    return torch.from_numpy(im_norm).float()


def tensor_to_numpy(x):
    """
    Converts a PyTorch tensor representing an image to a NumPy array.
    :param x: (tensor) PyTorch tensor.
    :return: (array) NumPy array.
    """
    return np.moveaxis(x.data.cpu().numpy(), 1, 3).squeeze()


def numpy_to_image(x):
    """
    Converts a NumPy array representing an image (normalized to [0, 1])
    to a PIL Image object.
    :param x: (array) NumPy array.
    :return: (object) PIL Image object.
    """
    return Image.fromarray(np.uint8(x * 255))


def pil_to_tensor(img):
    """
    Converts a PIL Image object to a PyTorch tensor.
    :param img: (object) PIL Image object.
    :return: (tensor) PyTorch tensor (normalized to [0, 1]).
    """
    img = np.asarray(img)
    im_norm = img / 255
    im_norm = np.moveaxis(im_norm[np.newaxis], 3, 1)
    return torch.from_numpy(im_norm).float()


class ProgressBar:
    """
    Creates a progress bar in the command line that illustrates
    the training progress. This progress bar was inspired by Keras,
    but the code was not taken from Keras.
    """

    def __init__(self, width=30):
        self.width = width

    def update(self, max_value, current_value, info):
        """
        Update the progress bar.
        :param max_value: (int) the maximum value of the progress bar.
        :param current_value: (int) the current value of the progress bar.
        :param info: (string) an info string that will be displayed to the right of the progress bar.
        """
        progress = int(round(self.width * current_value / max_value))
        bar = '=' * progress + '.' * (self.width - progress)
        prefix = '{}/{}'.format(current_value, max_value)

        prefix_max_len = len('{}/{}'.format(max_value, max_value))
        buffer = ' ' * (prefix_max_len - len(prefix))

        sys.stdout.write('\r {} {} [{}] - {}'.format(prefix, buffer, bar, info))
        sys.stdout.flush()

    def new_line(self):
        print()


