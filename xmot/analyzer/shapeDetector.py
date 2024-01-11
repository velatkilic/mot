import math
import cv2 as cv
from matplotlib.pyplot import draw
import numpy as np
from xmot.digraph.parser import load_blobs_from_text
from xmot.digraph.particle import Particle
from xmot.logger import Logger

LEGIT_CLOSED_CONTOUR_AREA_RATIO = 0.30     # A legit closed contour must take up at least this percentage of the bbox.
LEGIT_PARTICLE_AREA_RATIO = 0.15 # ?0.15
LEGIT_CONTOUR_AREA_MIN = 50   # Least permitted area for a contour to be considered as a contour
                              # of a solid particle. Contours below this value will be considered as 
                              # a partial boundary of a hollow shell.
CONFIDENT_PARTICLE_AREA = 100

BBOX_BUFFER_MIN = 2           # 2 pixels as buffer of bbox for contour detection.
BBOX_BUFFER_MAX = 5           # Max permitted value of buffer for expanding crop of particle to detect
                              # a valid contour. (Contour cannot be detected if the particle contact 
                              # the edge of crop of the img)
PADDING=1                     # White padding crop of particle with 2 pixels to make sure detected
                              # contours don't connect with the borders.


THRESHOLD_A2P_RATIO = 0.85      # Lower threshold of normalized Area-to-perimeter ratio for a shape
                                # to be considered as a circle. For perfect circle, a2p ratio is 1.
THRESHOLD_CIRCULAR_DEGREE = 0.5 # Threshold of param2 in Hough Circle transformation to consider
                                # detected circles as valid.

def detect_shape(particle: Particle, img, a2p_threshold=THRESHOLD_A2P_RATIO, padding=PADDING,
                 outdir=None, prefix=None) -> str:
    # Padding borders with white stripes to make sure at least one contour can be detected.
    img_crop = crop_particle(particle, img, buffer=0)

    # If particle width and height are 0, the padded area will be a (4 x 5) image.
    if img_crop.shape[0] * img_crop.shape[1] == 2*PADDING * (2*PADDING + 1):
        Logger.error("Cannot detect shape for particle with zero-sized bbox. " + 
                        "Frame: {:d}; ID: {:d}.".format(particle.get_time_frame(), particle.get_id()))
        shape = "undetermined_empty_crop"

    # Threshold
    blocksize = max(math.ceil(np.average(img_crop.shape)) // 2 * 2 + 1, 31) # Round to the next odd integer.
    img_threshed = adaptive_threshold(img_crop, cv.ADAPTIVE_THRESH_MEAN_C, blocksize = blocksize,
                                    offset = 2, is_grayscale = True)
    
    # Padding to prevent contour touching boundaries
    img_edited = cv.copyMakeBorder(img_threshed, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=255)

    # Morphological opening
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img_edited = cv.bitwise_not(img_edited)
    img_edited = cv.morphologyEx(img_edited, cv.MORPH_OPEN, kernel = kernel, iterations = 1)
    img_edited = cv.bitwise_not(img_edited)

    # Contour detection
    # Only the outmost contour matters to Shape detection.
    contours = detect_contours(img_edited, only_outmost=True)

    if outdir is not None:
        # Write intermediate pictures out for debugging.
        cv.imwrite(f"{outdir}/{prefix}_crop.png", img_crop)
        cv.imwrite(f"{outdir}/{prefix}_thresh.png", img_threshed)
        cv.imwrite(f"{outdir}/{prefix}_morph.png", img_edited)
        _img_contoured = cv.drawContours(cv.cvtColor(np.copy(img_edited), cv.COLOR_BGR2RGB), contours, -1,
                                         color=(0,0,255), thickness=1)
        cv.imwrite(f"{outdir}/{prefix}_contoured_len_{len(contours)}.png", _img_contoured)

    # There are only noises in the crop. The bbox might not enclosing any particle and is
    # fictitious from Kalman Filter. TODO: We need improve this.
    if len(contours) == 0:
        shape = "undetermined_no_particle"
        _img_debug = img_threshed
        if outdir is not None:
            cv.imwrite(f"{outdir}/{prefix}_decision_{particle.get_type()}_{particle.get_shape()}_{shape}.png", _img_debug)
        return shape
    
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
    cnt_areas = [] # list of contour areas
    for i in range(0, len(contours)):
        cnt_areas.append(cv.contourArea(contours[i]))
    cnt_max_area = contours[cnt_areas.index(max(cnt_areas))] # The single contour with the largest area.
    cnt_areas = np.array(cnt_areas)
    cnt_area_to_bbox_ratios = cnt_areas / particle.get_area_bbox()

    max_area = cnt_areas.max()
    max_cnt_to_bbox_ratio = cnt_area_to_bbox_ratios.max()
    
    if outdir is not None:
        _area_ratio = round(max_cnt_to_bbox_ratio, 3)
        _img_contoured = cv.drawContours(cv.cvtColor(np.copy(img_edited), cv.COLOR_BGR2RGB), [cnt_max_area], -1,
                                         color=(0,0,255), thickness=1)
        cv.imwrite(f"{outdir}/{prefix}_max_contour_areaRatio_{_area_ratio}.png", _img_contoured)

    shape = ""
    if (len(contours) == 1 and max_cnt_to_bbox_ratio >= LEGIT_PARTICLE_AREA_RATIO)  \
        or np.sum(cnt_area_to_bbox_ratios >= LEGIT_CLOSED_CONTOUR_AREA_RATIO) == 1 \
        or np.sum(cnt_areas >= CONFIDENT_PARTICLE_AREA) == 1 \
        or np.sum(cnt_area_to_bbox_ratios >= LEGIT_PARTICLE_AREA_RATIO) == 1:
        # There is one sinlge large contour in this crop.
        # The particle is a solid particle. Use normalized area-to-perimeter ratio to
        # determine shape.
        perimeter = cv.arcLength(cnt_max_area, True)
        a2p_ratio = 4 * math.pi * max_area / (perimeter ** 2)
        if outdir is not None:
            cv.imwrite(f"{outdir}/{prefix}_max_contour_a2p_{round(a2p_ratio, 2)}.png", _img_contoured)
        if a2p_ratio >= a2p_threshold:
            shape = "circle"
            _img_debug = np.copy(img_edited)
        else:
            shape = "non-circle" # Could be agglomerate.
            _img_debug = np.copy(img_edited)
    elif np.sum(cnt_areas >= CONFIDENT_PARTICLE_AREA) >= 2 \
            or np.sum(cnt_area_to_bbox_ratios >= LEGIT_CLOSED_CONTOUR_AREA_RATIO) >= 2 \
            or np.sum(cnt_area_to_bbox_ratios >= LEGIT_PARTICLE_AREA_RATIO) >= 2:
        # Multiple large particles or large ratio contours in this image
        # Very likely to be agglomoerates of particles with disconnected parts
        _img_debug = np.copy(img_edited)
        shape = "non-circle"
    elif len(contours) >= 1 and max_cnt_to_bbox_ratio < LEGIT_PARTICLE_AREA_RATIO:
        # There are multiple contours and all of them have very small area, suggesting they are 
        # contours of broken boundaries of hollow shells, instead of solid particles. Therefore,
        # use Hough circle transformation to determine shape.
        #
        # In rare cases, the particle could be agglomerates or solid particles with high
        # aspect-ratio, and the rectangular bounding box contains large empty area, resulting
        # in a low contour-image ratio. But Hough circle should still be capable of determing
        # them as non-circle becuase of the high-aspect ratio.
        # Hough circle transform
        _canny_threshold = 30
        _accumulator_threshold = 20
        #_circular_degree = 0.1
        #_img_to_detect = img_edited # Strategy 1: black shell edge, white background
        #_img_to_detect = cv.bitwise_not(img_edited) # Strategy 2: white shell edge, black background
        _img_to_detect = cv.bitwise_not(img_threshed) # Strategy 3: Use threshed picture, no morph operations.
        while _accumulator_threshold >= 10:
        #while(circular_degree >= 0.2):
            circles = cv.HoughCircles(_img_to_detect, cv.HOUGH_GRADIENT, 1, max(img_crop.shape)/10,
                                      param1=_canny_threshold, param2=_accumulator_threshold,
                                      minRadius = math.floor(min(img_crop.shape)/4), 
                                      maxRadius = min(img_crop.shape))
            #circles = cv.HoughCircles(_img_to_detect, cv.HOUGH_GRADIENT_ALT, 1, max(img_crop.shape)/10,
            #                          param1=_canny_threshold, param2=_circular_degree,
            #                          minRadius = math.floor(min(img_crop.shape)/4), 
            #                          maxRadius = min(img_crop.shape[0]))
            if circles is not None:
                break
            _accumulator_threshold -= 1
            #_circular_degree -= 0.05

        if outdir is not None:
            _img_hough = cv.cvtColor(np.copy(img_edited), cv.COLOR_BGR2RGB)
            _img_hough2 = cv.cvtColor(np.copy(_img_to_detect), cv.COLOR_BGR2RGB)
            if circles is not None:
                # The first is the best fitted one. So only draw the first.
                _max_circle = circles[0, 0]
                _max_circle = np.uint16(np.around(_max_circle))
                cv.circle(_img_hough, (_max_circle[0], _max_circle[1]), _max_circle[2], (0, 255, 0), 1)
                cv.circle(_img_hough, (_max_circle[0], _max_circle[1]), 1, (0, 0, 255), -1) # center
                cv.circle(_img_hough2, (_max_circle[0], _max_circle[1]), _max_circle[2], (0, 255, 0), 1)
                cv.circle(_img_hough2, (_max_circle[0], _max_circle[1]), 1, (0, 0, 255), -1) # center

            cv.imwrite(f"{outdir}/{prefix}_hough_{_accumulator_threshold}.png", _img_hough)
            cv.imwrite(f"{outdir}/{prefix}_hough_threshed_{_accumulator_threshold}.png", _img_hough2)
            #cv.imwrite(f"{outdir}/{prefix}_hough_{round(circular_degree,2)}.png", _img_hough)
            #cv.imwrite(f"{outdir}/{prefix}_hough_threshed_{round(circular_degree, 2)}.png", _img_hough2)

        
        #if circular_degree >= THRESHOLD_CIRCULAR_DEGREE:
        #    Logger.debug("Circular degree in Hough for detecting a circle in this hollow sheel "
        #                    "is {:.2f}".format(circular_degree))
        #    return "circle"
        if circles is not None:
            _img_debug = np.copy(img_crop)
            shape = "circle"
        else:
            _img_debug = np.copy(img_crop)
            shape = "non-circle" # Cannot find a circle with permitted circular degree larger than
                                 # THRESHOLD_CIRCULAR_DEGREE. It's non-circle.
    # else: len(contours) > 1 and max_area / particle.get_area_bbox() between LEGIT_PARTICLE_AREA_RATIO and LEGIT_CLOSED_CONTOUR_RATIO
                
    # 1. no contour can be detected.

    if shape == "":
        _img_debug = np.copy(img_crop)
        if particle.get_area() < CONFIDENT_PARTICLE_AREA:
            shape = "undetermined_too_small"
        else:
            shape = "undetermined"
    
    if outdir is not None:
        cv.imwrite(f"{outdir}/{prefix}_decision_{particle.get_type()}_{particle.get_shape()}_{shape}.png", _img_debug)
    return shape

def detect_contours(img, is_grayscale = True, only_outmost=False):
    """
    Note:
    1. cv.findContours cannot detect contours for sections connected to the edge of the image.
       So BOX_BUFFER is important.

    Attribute:
        img          np.ndarray     Input image in Opencv format (i.e numpy.ndarray)
        only_outmost bool           Use the cv.RETR_OUT

    Return:
        contour
        Array of [x, y, w, h] defining the bounding rectangulars.
    """
    list_bbox = []
    if not is_grayscale:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.bitwise_not(np.copy(img))  # Particles will appear as white and its boundary being detected.
                                        # The image boundary won't be a contour in this case.

    if only_outmost:
        contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    else:
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = list(contours)
    #contours.pop(0)  # The first contour is always the entire image. Remove.
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

def crop_particle(particle: Particle, img, buffer = 1):
    """
    Args:
        particle    Particle
        img         ndarray     Entire frame of video containing the input particle
        is_binary   boolean     Whether argument image is binary
        buffer      int         When positive, add buffer on all four borders of the crop to make sure the whole
                                particle is enclosed in the bounding box.

    Crop particle according to its bbox location and size. Add buffers to its bbox to
    aid following identification. Note that negative coordinates will wrap around as what python
    slicing does, so we need to check whether they are out of the image before cropping.
    """
    x1, y1, x2, y2 = particle.get_bbox_torch()
    
    # Coordinates of upper left corner of the crop
    # Add buffer to the bbox to make sure the entire 
    # particle has been enclosed in the bounding
    # box.
    x1 = x1 - buffer if x1 - buffer >= 0 else 0
    y1 = y1 - buffer if y1 - buffer >= 0 else 0

    # Coordinates of lower right corner of the crop
    x2 = x2 + buffer if x2 + buffer <= img.shape[1] else img.shape[1]
    y2 = y2 + buffer if y2 + buffer <= img.shape[0] else img.shape[0] 
    
    if len(img.shape) == 2:
        img_crop = img[y1:y2, x1:x2]
    else:
        # First index is row, which is y in the x-y coordinate sense!
        img_crop = img[y1:y2, x1:x2, :]

    return img_crop