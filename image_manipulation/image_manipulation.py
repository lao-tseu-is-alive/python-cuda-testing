import PIL
import numpy as np
import cupy as cp
from PIL import Image
from pathlib import Path
import sys


def read_pil_image(image_path):
    if Path(image_path).is_file():
        try:
            img = Image.open(image_path, mode="r")
        except PIL.UnidentifiedImageError as e:
            print(f"Error in file {image_path}: {e}")
            sys.exit(1)
    else:
        print(format("file {%s} does not exist", image_path))
        sys.exit(1)
    return img


def get_numpy_arr_from_pil_img(img_pil_ref):
    return np.asarray(img_pil_ref)


def get_cupy_arr_from_pil_img(img_pil_ref):
    return cp.asarray(img_pil_ref)


if __name__ == '__main__':
    img_path = "../data/nature.jpg"
    if len(sys.argv) > 1:
        img_path = sys.argv[0]
    print("Trying to load image :{:s}".format(img_path))
    im = read_pil_image(img_path)
    print("Image format: {:s}".format(im.format))
    print("Image size  Width x Height : {}".format(im.size))
    im.show(title="original")

    print("## converting to numpy array")
    a_np = get_numpy_arr_from_pil_img(im)
    print("image numpy array shape {}".format(a_np.shape))
    print("## converting to cupy array")
    a_cp = get_cupy_arr_from_pil_img(im)
    print("image cupy array shape {}".format(a_cp.shape))

    print("image cupy array keeping only the red channel")
    red_cp = cp.copy(a_cp)
    red_cp[:, :, 1] = 0
    red_cp[:, :, 2] = 0
    print("converting back to PIL image")
    im_red = Image.fromarray(cp.asnumpy(red_cp))
    im_red.show("red component of image only")

    print("image cupy array keeping only the green channel")
    green_cp = cp.copy(a_cp)
    green_cp[:, :, 0] = 0
    green_cp[:, :, 2] = 0
    print("converting back to PIL image")
    im_green = Image.fromarray(cp.asnumpy(green_cp))
    im_green.show("green component of image only")

    print("image cupy array keeping only the blue channel")
    blue_cp = cp.copy(a_cp)
    blue_cp[:, :, 0] = 0
    blue_cp[:, :, 1] = 0
    print("converting back to PIL image")
    im_blue = Image.fromarray(cp.asnumpy(blue_cp))
    im_blue.show("blue component of image only")

