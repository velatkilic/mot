import math
import cv2 as cv
from matplotlib.pyplot import draw
import numpy as np
from xmot.digraph.utils import load_blobs_from_text
from xmot.digraph.particle import Particle
from xmot.logger import Logger

LEGIT_CLOSED_CONTOUR_AREA_RATIO = 0.30     # A legit closed contour must take up at least 50% of the crop area.
LEGIT_CONTOUR_AREA_MIN = 64   # Least permitted area for a contour to be considered as a contour
                              # of a solid particle, instead of contour of a partial boundary
                              # of a hollow shell.
BBOX_BUFFER_MIN = 2           # 2 pixels as buffer of bbox for contour detection.
BBOX_BUFFER_MAX = 5           # Max permitted value of buffer for expanding crop of particle to detect
                              # a valid contour. (Contour cannot be detected if the particle contact 
                              # the edge of crop of the img)
THRESHOLD_A2P_RATIO = 0.9     # Lower threshold of normalized Area-to-perimeter ratio for a shape
                              # to be considered as a circle. For perfect circle, a2p ratio is 1.
THRESHOLD_CIRCULAR_DEGREE = 0.5 # Threshold of param2 in Hough Circle transformation to consider
                                # detected circles as valid.


def detect_shape(self, particle: Particle, img) -> str:
    buffer = BBOX_BUFFER_MIN
    while (buffer < BBOX_BUFFER_MAX):
        img_crop = crop_particle(particle, img, buffer)
        if img_crop.shape[0] * img_crop.shape[1] == 0: # empty image
            Logger.error("Cannot detect shape for particle with zero-sized bbox. " + 
                         "Frame: {:d}; ID: {:d}.".format(particle.get_time_frame(), particle.get_id()))
            shape = "undetermined"

        # Threshold
        img_edited = adaptive_threshold(img_crop, cv.ADAPTIVE_THRESH_MEAN_C,
                                        blocksize = 31, offset = 2, is_grayscale = True)
        
        # Morphological opening
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        img_edited = cv.bitwise_not(img_edited)
        img_edited = cv.morphologyEx(img_edited, cv.MORPH_OPEN, kernel = kernel, iterations = 1)
        img_edited = cv.bitwise_not(img_edited)

        # Contour detection
        contours = detect_contours(img_edited)
        
        # Determine whether the particle is a solid particle or a hollow shell.
        legit_contours = []
        for cnt in contours:
            if cv.contourArea(cnt) < LEGIT_CONTOUR_AREA_MIN:
                continue
            legit_contours.append(cnt)
        
        if len(legit_contours) == 0:
            buffer += 1
            continue # No legit contours for following operation.

        # For particles with bubbles, there could be mulitple contours. Therefore, use the
        # largest contour to determine shape.
        max_cnt_area = 0
        max_area_cnt = None
        for cnt in legit_contours:
            if cv.contourArea(cnt) > max_cnt_area:
                max_cnt_area = cv.contourArea(cnt)
                max_area_cnt = cnt
        
        if float(max_cnt_area) / (img_crop.shape[0] * img_crop.shape[1]) >= LEGIT_CLOSED_CONTOUR_AREA_RATIO:
            # The particle is a solid particle. Use normalized area-to-perimeter ratio to
            # determine shape.
            perimeter = cv.arcLength(max_area_cnt, True)
            a2p_ratio = 4 * math.pi * max_cnt_area / (perimeter ** 2)
            if a2p_ratio > THRESHOLD_A2P_RATIO:
                return "circle"
            else:
                return "non-circle"
        else:
            # Contours all have very small area, suggesting they are contours of broken
            # boundaries of hollow shells, instead of solid particles. Use Hough circle transformation
            # to determine shape.
            #
            # In rare cases, the particle could be agglomerates or solid particles with high
            # aspect-ratio, and the rectangular bounding box contains large empty area, resulting
            # in a low contour-image ratio. But Hough circle should still be capable of determing
            # them as non-circle becuase of the high-aspect ratio.
            # Hough circle transform
            circular_degree = 0.9
            canny_threshold = 90
            while(circular_degree >= THRESHOLD_CIRCULAR_DEGREE):
                circles = cv.HoughCircles(img_edited, cv.HOUGH_GRADIENT, 1, img_crop.shape[0]/10,
                                          param1=canny_threshold, param2=circular_degree,
                                          minRadius = math.floor(img_crop.shape[0]/4), 
                                          maxRadius = math.ceil(img_crop.shape[0]/2))
                if circles is not None:
                    Logger.debug("Circular degree in Hough for detecting a circle in this hollow sheel "
                                 "is {:.2f}".format(circular_degree))
                    return "circle"
                circular_degree -= 0.05
            
            return "non-circle" # Cannot find a circle with permitted circular degree larger than
                                # THRESHOLD_CIRCULAR_DEGREE. It's non-circle.
                
    # Most likely the particle is on the edge of the video frame and no contour can be detected.
    return "undetermined"

def detect_contours(img, is_grayscale = True):
    """
    Note:
    1. cv.findContours cannot detect contours for sections connected to the edge of the image.
       So BOX_BUFFER is important.

    Attribute:
        img Input image in Opencv format (i.e numpy.ndarray)

    Return:
        contour
        Array of [x, y, w, h] defining the bounding rectangulars.
    """
    list_bbox = []
    if not is_grayscale:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.pop(0)  # The first contour is the entire image. Remove.
    return contours

def binary_threshold(img, threshold = 90, is_grayscale = False):
    if not is_grayscale:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, img_threshold = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return img_threshold

def adaptive_threshold(img, method = cv.ADAPTIVE_THRESH_MEAN_C, blocksize = 15, offset = 0, 
                        is_grayscale = False):
    
    if not is_grayscale:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if img is None or len(img) == 0:
        Logger.error("adaptive_threshold: Image is empty. Please check.")
    img_threshold = cv.adaptiveThreshold(img, 255, method, cv.THRESH_BINARY, blocksize, offset)
    return img_threshold

def crop_particle(particle, img, is_grayscale = False, buffer = 2):
    """
    Args:
        particle    Particle \n
        img         ndarray     Entire frame of video containing the input particle\n
        is_binary   boolean     Whether argument image is binary\n

    Crop particle according to its bbox location and size. Add buffers to its bbox to
    aid following identification.
    """
    x, y = particle.get_position()
    w, h = particle.get_bbox()
    
    # Coordinates of upper left corner of the crop
    # Add buffer to the bbox to make sure the entire particle has been enclosed in the bounding
    # box.
    x1 = x - buffer if x - buffer >= 0 else 0
    y1 = y - buffer if y - buffer >= 0 else 0

    # Coordinates of lower right corner of the crop
    x2 = x + w + buffer \
        if x + w + buffer <= img.shape[1] else img.shape[1]
    y2 = y + h + buffer \
        if y + h + buffer <= img.shape[0] else img.shape[0] 
    if is_grayscale:
        cropped_img = img[y1:y2, x1:x2]
    else:
        # First index is row, which is y in the x-y coordinate sense!
        cropped_img = img[y1:y2, x1:x2, :]
    return cropped_img