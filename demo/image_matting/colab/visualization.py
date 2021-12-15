import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet
from common import common_path

import numpy as np
from PIL import Image


def combined_display(image, matte):
    # calculate display resolution
    w, h = image.width, image.height
    rw, rh = 800, int(h * 800 / (3 * w))

    # obtain predicted foreground
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0:3]
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

    # combine image, foreground, and alpha into one line
    combined = np.concatenate((image, foreground, matte * 255), axis=1)
    combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))
    return combined


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str,
                        default=os.path.join(common_path.project_dir, 'demo/image_matting/colab/input'),
                        help='path of input images')
    parser.add_argument('--output-path', type=str,
                        default=os.path.join(common_path.project_dir, 'demo/image_matting/colab/output'),
                        help='path of output images')
    parser.add_argument('--ckpt-path', type=str,
                        default=os.path.join(common_path.project_dir,
                                             'pretrained/modnet_photographic_portrait_matting.ckpt'),
                        help='path of pre-trained MODNet')
    args = parser.parse_args()
    # visualize all images
    image_names = os.listdir(args.input_path)
    for image_name in image_names:
        matte_name = image_name.split('.')[0] + '.png'
        image = Image.open(os.path.join(args.input_path, image_name))
        matte = Image.open(os.path.join(args.output_path, matte_name))
        # display(combined_display(image, matte))
        combined_display(image, matte).show()
        print(image_name, '\n')