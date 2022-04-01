import sys
sys.path.append("/mnt/d/JHU/Research/Machine_Learning_Characterization/mot/")

import os
import cv2 as cv
from src.analyzer.shapeDetector import ShapeDetector
from src.digraph.utils import load_blobs_from_text
from src.digraph.particle import Particle

# Script
particles = load_blobs_from_text("./0/blobs.txt")
img = cv.imread("./0/orig_0.jpg")  # 3-d array of shape (640, 624, 3)
#img = binary_threshold(img)
img = img[0:624, 0:624]  # use the sample of the size 624x624.
detector = ShapeDetector()
for particle in particles:
    shape = detector.detect_shape(particle, img)
    print(shape)
