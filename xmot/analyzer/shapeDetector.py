import math
import cv2 as cv
from matplotlib.pyplot import draw
import numpy as np
from xmot.digraph.parser import load_blobs_from_text
from xmot.digraph.particle import Particle
from xmot.logger import Logger

LEGIT_CLOSED_CONTOUR_AREA_RATIO = 0.30     # A legit closed contour must take up at least this percentage of the bbox.
LEGIT_PARTICLE_AREA_RATIO = 0.20
LEGIT_CONTOUR_AREA_MIN = 50   # Least permitted area for a contour to be considered as a contour
                              # of a solid particle. Contours below this value will be considered as 
                              # a partial boundary of a hollow shell.
CONFIDENT_PARTICLE_AREA = 100

BBOX_BUFFER_MIN = 2           # 2 pixels as buffer of bbox for contour detection.
BBOX_BUFFER_MAX = 5           # Max permitted value of buffer for expanding crop of particle to detect
                              # a valid contour. (Contour cannot be detected if the particle contact 
                              # the edge of crop of the img)
PADDING=2                     # White padding crop of particle with 2 pixels to make sure detected
                              # contours don't connect with the borders.


THRESHOLD_A2P_RATIO = 0.85      # Lower threshold of normalized Area-to-perimeter ratio for a shape
                                # to be considered as a circle. For perfect circle, a2p ratio is 1.
THRESHOLD_CIRCULAR_DEGREE = 0.5 # Threshold of param2 in Hough Circle transformation to consider
                                # detected circles as valid.

def detect_shape(particle: Particle, img) -> str:
    # Padding borders with white stripes to make sure at least one contour can be detected.
    img_crop = crop_particle(particle, img, padding=PADDING) 

    # If particle width and height are 0, the padded area will be a (4 x 5) image.
    if img_crop.shape[0] * img_crop.shape[1] == 2*PADDING * (2*PADDING + 1):
        Logger.error("Cannot detect shape for particle with zero-sized bbox. " + 
                        "Frame: {:d}; ID: {:d}.".format(particle.get_time_frame(), particle.get_id()))
        shape = "undetermined_empty_crop"

    # Threshold
    blocksize = math.ceil(np.average(img_crop.shape)) // 2 * 2 + 1 # Round to the next odd integer.
    img_edited = adaptive_threshold(img_crop, cv.ADAPTIVE_THRESH_MEAN_C, blocksize = blocksize,
                                    offset = 2, is_grayscale = True)
    
    # Morphological opening
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img_edited = cv.bitwise_not(img_edited)
    img_edited = cv.morphologyEx(img_edited, cv.MORPH_OPEN, kernel = kernel, iterations = 1)
    img_edited = cv.bitwise_not(img_edited)

    # Contour detection
    contours = detect_contours(img_edited)

    # There are only noises in the crop. The bbox might not enclosing any particle and is
    # fictitious from Kalman Filter. TODO: We need improve this.
    if len(contours) == 0:
        return "undertermined_no_particle"
    
    # Determine whether the particle is a solid particle or a hollow shell.
    ## Graduately increase buffer until legit contours are found.
    #legit_contours = []
    #for cnt in contours:
    #    if cv.contourArea(cnt) < LEGIT_CONTOUR_AREA_MIN:
    #        continue
    #    legit_contours.append(cnt)
    #
    #if len(legit_contours) == 0:
    #    buffer += 1
    #    continue # No legit contours for following operation.

    # For particles with bubbles, there could be mulitple contours. Therefore, use the
    # largest contour to determine shape.
    cnt_areas = {} # index: area
    for i in range(0, len(contours)):
        cnt_areas[i] = cv.contourArea(contours[i])
    cnt_max_area = contours[max(cnt_areas, key=cnt_areas.get)] # Contour with the largest area.
    max_area = float(max(cnt_areas.values()))
    
    if len(contours) == 1 or max_area / particle.get_area_bbox() >= LEGIT_CLOSED_CONTOUR_AREA_RATIO:
        # The particle is a solid particle. Use normalized area-to-perimeter ratio to
        # determine shape.
        perimeter = cv.arcLength(cnt_max_area, True)
        a2p_ratio = 4 * math.pi * max_area / (perimeter ** 2)
        if a2p_ratio > THRESHOLD_A2P_RATIO:
            return "circle"
        else:
            return "non-circle" # Could be agglomerate.
    elif len(contours) > 1 and max_area / particle.get_area_bbox() < LEGIT_PARTICLE_AREA_RATIO:
        # There are multiple contours and all of them have very small area, suggesting they are 
        # contours of broken boundaries of hollow shells, instead of solid particles. Therefore,
        # use Hough circle transformation to determine shape.
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
    # else: len(contours) > 1 and max_area / particle.get_area_bbox() between LEGIT_PARTICLE_AREA_RATIO and LEGIT_CLOSED_CONTOUR_RATIO
                
    # 1. no contour can be detected.
    if particle.get_area() < CONFIDENT_PARTICLE_AREA:
        return "undertermined_too_small"
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
    contours.pop(0)  # The first contour is always the entire image. Remove.
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

def crop_particle(particle, img, is_grayscale = False, padding = 1, buffer = -1):
    """
    Args:
        particle    Particle
        img         ndarray     Entire frame of video containing the input particle
        is_binary   boolean     Whether argument image is binary
        padding     int         When positive, padding the cropped image with white borders to make sure at least one
                                contour can be detected in the crop.
        buffer      int         When positive, add buffer on all four borders of the crop to make sure the whole
                                particle is enclosed in the bounding box.

    Crop particle according to its bbox location and size. Add buffers to its bbox to
    aid following identification. Note that negative coordinates will wrap around as what python
    slicing does, so we need to check whether they are out of the image before cropping.
    """
    x, y = particle.get_position()
    w, h = particle.get_bbox()
    
    if padding > 0:

        img_crop = img[y:y + h, x:x + w]
        if is_grayscale:
            # Padding all four borders equally.
            img_crop = cv.copyMakeBorder(img_crop, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=255)
        else:
            img_crop = cv.copyMakeBorder(img_crop, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=[255, 255, 255])
    else:

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
            img_crop = img[y1:y2, x1:x2]
        else:
            # First index is row, which is y in the x-y coordinate sense!
            img_crop = img[y1:y2, x1:x2, :]

    return img_crop