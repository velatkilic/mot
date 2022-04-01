import sys
sys.path.append("/mnt/d/JHU/Research/Machine_Learning_Characterization/mot/")

import os
import cv2 as cv
import numpy as np
import math
from src.analyzer.shapeDetector import ShapeDetector
from src.digraph.utils import load_blobs_from_text
from src.digraph.particle import Particle

def combine_images(n_row, n_column, images):
    """
    Paste a list of images into a panel of n_row * n_column. Assume all images in the list
    share the same size of the first image of the list.
    """
    if len(images[0].shape) == 3:
        h0, w0, n_color = images[0].shape
    elif len(images[0].shape) == 2:
        h0, w0 = images[0].shape
    img_combined = np.zeros((h0 * n_row, w0 * n_column), np.uint8)

    for i in range(0, len(images)):
        img  = images[i]
        row = math.floor(i / n_column)
        column = i % n_column
        #if row == 0:
        img_combined[(h0*row):(h0*(row+1)), (w0*column):(w0*(column+1))] = img

    return img_combined

# Script
particles = load_blobs_from_text("../0/blobs.txt")
img = cv.imread("../0/orig_0.jpg", cv.IMREAD_GRAYSCALE)  # 3-d array of shape (640, 624, 3)
img = img[0:624, 0:624]  # use the sample of the size 624x624.

for particle in particles:
    id = particle.get_id()
    os.makedirs("./{:d}".format(id), exist_ok=True)
    img_crop = ShapeDetector.crop_particle(particle, img, True)
    cv.imwrite("./{:d}/orig_p_{:d}.jpg".format(id, id), img_crop)
    
    # Binary
    thresholds = list(range(75, 101, 5))
    images_binary = []
    for i in range(0, len(thresholds)):
        img_binary_threshold = ShapeDetector.binary_threshold(img_crop, thresholds[i], True)
        images_binary.append(img_binary_threshold)
        #cv.imwrite("./{:d}/binary_{:d}.jpg".format(id, thresholds[i]), img_binary_threshold)
    img_binary_combined = combine_images(1, len(thresholds), images_binary)
    cv.imwrite("./{:d}/binary_{:d}.jpg".format(id, id), img_binary_combined)

    # Adaptive Mean
    offsets = list(range(0, 6))
    blocksizes = list(range(7, 32, 4))
    
    images_adaptive_mean = []
    for i in range(0, len(offsets)):
        for j in range(0, len(blocksizes)):
            img_adaptive_mean = ShapeDetector.adaptive_threshold(img_crop, cv.ADAPTIVE_THRESH_MEAN_C,
                blocksize = blocksizes[j], offset = offsets[i], is_grayscale = True)
            #cv.imwrite("./{:d}/adaptive_mean_{:d}_{:d}.jpg".format(id, blocksizes[j], offsets[i]), 
            #    img_adaptive_mean)
            images_adaptive_mean.append(img_adaptive_mean)
    img_adaptive_mean_combined = combine_images(len(offsets), len(blocksizes), images_adaptive_mean)
    cv.imwrite("./{:d}/adaptive_mean_{:d}.jpg".format(id, id), img_adaptive_mean_combined)
    
    
    # Adaptive Gaussian
    images_adaptive_gaussian = []
    for i in range(0, len(offsets)):
        for j in range(0, len(blocksizes)):
            img_adaptive_gaussian = ShapeDetector.adaptive_threshold(img_crop, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                blocksize = blocksizes[j], offset = offsets[i], is_grayscale = True)
            #cv.imwrite("./{:d}/adaptive_gaussian_{:d}_{:d}.jpg".format(id, blocksizes[j], offsets[i]),
            #    img_adaptive_gaussian)
            images_adaptive_gaussian.append(img_adaptive_gaussian)
    img_adaptive_gaussian_combined = combine_images(len(offsets), len(blocksizes), images_adaptive_gaussian)
    cv.imwrite("./{:d}/adaptive_gaussian_{:d}.jpg".format(id, id), img_adaptive_gaussian_combined)

