import sys, os
import cv2 as cv
import numpy as np
import natsort, re
import math
from pathlib import Path
from scipy.stats import mode
from typing import List, Tuple

from xmot.config import IMAGE_FORMAT, IMAGE_FILE_PATTERN

def subtract_brightfield_by_scaling(img_video, img_bf, scale = 0.8, shift_back=False):
    """
    Deprecated. Leave here as a record.

    Subtract IMG_BF from IMG_VIDEO and return a new image.
    """
    img_bf_invert = cv.bitwise_not(img_bf)
    img_video_invert = cv.bitwise_not(img_video)
    bf_mode = mode(img_bf_invert, axis=None)[0][0]
    video_mode = mode(img_video_invert, axis=None)[0][0]
    factor = scale * video_mode / bf_mode # Make the peak of histogram of the brightfield 
                                          # 0.8 to that of the video.
    # Scaling the peak of the pixel distribution of the brightfield image relative to that
    # of the video image.
    img_bf_invert = np.array(img_bf_invert * factor, dtype=np.uint8)

    # Taking care of the underflow problem of uint8.
    img_inverted_subtract = np.subtract(img_video_invert, img_bf_invert)
    img_inverted_subtract[img_bf_invert > img_video_invert] = 0

    img_inverted_subtract = cv.bitwise_not(img_inverted_subtract) # Inverse back so particles are dark.

    if shift_back:
        _mode_orig = mode(img_video, axis=None)[0][0]
        _mode_result = mode(img_inverted_subtract, axis=None)[0][0]
        img_inverted_subtract = img_inverted_subtract - (_mode_result - _mode_orig)

    return img_inverted_subtract, img_video_invert, img_bf_invert, bf_mode, video_mode, factor

def subtract_brightfield_by_shifting(img_video, img_bf, scale=1, shift_back=False):
    """
    Subtract IMG_BF from IMG_VIDEO and return a new image.

    This function shifts the brightfield image rather than scaling it by multiple a factor. This
    way the shape of the pixcel distribution is saved.
    """
    img_bf_invert = cv.bitwise_not(img_bf)
    img_video_invert = cv.bitwise_not(img_video)
    bf_invert_mode = mode(img_bf_invert, axis=None)[0][0]
    video_invert_mode = mode(img_video_invert, axis=None)[0][0]
    
    # Shfit the peak of the pixel distribution of bf image to align with that of the video.
    shift = bf_invert_mode - scale * video_invert_mode
    img_bf_invert_shifted = np.array(img_bf_invert - shift, dtype=np.uint8)
    img_bf_invert_shifted[img_bf_invert < shift] = 0 # Taking care of the underflow problem of uint8.

    img_inverted_subtract = np.subtract(img_video_invert, img_bf_invert_shifted)
    img_inverted_subtract[img_video_invert < img_bf_invert_shifted] = 0

    img_result = cv.bitwise_not(img_inverted_subtract) # Inverse back so particles are dark.

    if shift_back:
        # Shift the pixel distribution of image back to the original peak.
        _video_mode = mode(img_result, axis=None)[0][0] # likely a large value close to 256.
        _orig_video_mode = mode(img_video, axis=None)[0][0]
        _shift = _video_mode - _orig_video_mode
        # Recover the original background pixel value by subtracting a constant value
        img_result = img_result - _shift

    return img_result, img_video_invert, img_bf_invert_shifted, bf_invert_mode, video_invert_mode, shift

def get_contour_center(cnt) -> List[int]:
    moments = cv.moments(cnt) # OpenCV contour object: numpy.ndarray of shape (n, 1, 2)
    center_x = int(moments["m10"] / moments['m00'])
    center_y = int(moments["m01"] / moments['m00'])
    return [center_x, center_y]

def load_images_from_dir(dir, start_id=0, end_id=sys.maxsize, ext=None, grayscale=True) \
    -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all images from DIR, return the images and corresponding image file names in two lists.

    Filter the images by id if START_ID and END_ID are given. If EXT is None, use the
    extension of the first legit image file.

    TODO: Refactor to use imageio.get_reader(). Don't reinvent wheel.
    """
    if ext == None:
        files = [os.path.join(dir, f) for f in os.listdir(dir)]
        files = [f for f in files if os.path.isfile(f)]
        for f in files:
            if f.split(".")[-1] in IMAGE_FORMAT:
                ext = f.split(".")[-1]
                break
        files = [f for f in files if f.endswith(ext)]
    else:
        files = [f for f in os.listdir(dir) if f.endswith(ext)]
    
    files = natsort.natsorted(files)
    #files.sort(key=lambda f: int(re.match(".*_([a-zA-Z]*)([0-9]+)\.([a-z]+)", f).group(2)))
    if re.match(IMAGE_FILE_PATTERN, files[0]) is not None:
        files = [f for f in files if start_id <= int(re.match(IMAGE_FILE_PATTERN, f).group(3)) <= end_id]
    else:
        # The images might not contain a video id. Use a shorter regular expression.
        files = [f for f in files if start_id <= int(re.match(".*_([a-zA-Z]*)([0-9]+)\.([a-zA-Z]+)", f).group(2)) <= end_id]


    if len(files) == 0:
        print(f"No valid image files found in {dir} with extension {ext}")

    if grayscale:
        orig_images = [cv.imread(f, cv.IMREAD_GRAYSCALE) for f in files]
    else:
        orig_images = [cv.imread(f) for f in files]  # color pics are already in BGR order, not RBG
    orig_image_names = [Path(f).resolve().name for f in files]
    return orig_images, orig_image_names

def combine_images(n_row, n_column, images):
    """
    Paste a list of images into a panel of n_row * n_column. Assume all images in the list
    share the same size of the first image of the list.
    """
    if len(images[0].shape) == 3:
        h0, w0, n_color = images[0].shape
        img_combined = np.zeros((h0 * n_row, w0 * n_column, n_color), np.uint8)
    elif len(images[0].shape) == 2:
        h0, w0 = images[0].shape
        img_combined = np.zeros((h0 * n_row, w0 * n_column), np.uint8)

    for i in range(0, len(images)):
        img  = images[i]
        row = math.floor(i / n_column)
        column = i % n_column
        if len(images[0].shape) == 3:
            img_combined[(h0*row):(h0*(row+1)), (w0*column):(w0*(column+1)), :] = img
        else:
            img_combined[(h0*row):(h0*(row+1)), (w0*column):(w0*(column+1))] = img

    return img_combined