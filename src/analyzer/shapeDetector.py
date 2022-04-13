import math
import cv2 as cv
from matplotlib.pyplot import draw
import numpy as np
from src.digraph.utils import load_blobs_from_text
from src.digraph.particle import Particle
from src.logger import Logger

class ShapeDetector:

    AREA_THRESHOLD_LOWER = 20 # Bounding boxes need to have at least an area of 
                              # 20 pixel^2 to be a particle.
    AREA_THRESHOLD_UPPER = 100000
    P2A_RATIO_THRESHOLD = 4
    BBOX_BUFFER = 5           # 3 pixels as buffer of bbox for contour detection.

    def __init__(self, buffer = 10):
        # TODO: define default configurations here.
        self.buffer = buffer
        pass

    def detect_shape(self, particle: Particle, img) -> str:
        x, y = particle.get_position()
        w, h = particle.get_bbox()

        #print(x, y)
        # 1. Cut image out
        # TODO: add a loop to gradually enlarge the bbox.
        
        # Coordinates of upper left corner of the crop
        x1 = x - ShapeDetector.BBOX_BUFFER if x - ShapeDetector.BBOX_BUFFER >= 0 else 0
        y1 = y - ShapeDetector.BBOX_BUFFER if y - ShapeDetector.BBOX_BUFFER >= 0 else 0

        # Coordinates of lower right corner of the crop
        x2 = x + w + ShapeDetector.BBOX_BUFFER \
            if x + w + ShapeDetector.BBOX_BUFFER <= img.shape[0] else img.shape[0]
        y2 = y + h + ShapeDetector.BBOX_BUFFER \
            if y + h + ShapeDetector.BBOX_BUFFER <= img.shape[1] else img.shape[1] 
        cropped_img = img[y1:y2, x1:x2, :]  # First index is row, which is y in the x-y coordinate sense!
        
        # Threshold

        # Canny

        # Hough circle transform

        cropped_img_binary = ShapeDetector.binary_threshold(cropped_img)
        
        # 2. Get contour
        cv.imwrite("./p_{:d}.jpg".format(particle.get_id()), cropped_img_binary)
        contour, bbox = ShapeDetector.detect_contour(cropped_img_binary)
        cv.drawContours(cropped_img, contour, -1, [0, 0, 255], 1)
        cv.imwrite("./p_{:d}_with_contour.jpg".format(particle.get_id()), cropped_img)
        
        # 3. Calculate ratio of contour length and area
        #if len(contours) > 1:
        #    print("Number of contours in the cropped image are: " + str(len(contours)))
        perimeter = cv.arcLength(contour, True)
        area = cv.contourArea(contour)
        p2a_ratio = perimeter ** 2 / area  # Take square to normalize
        print(p2a_ratio)
        shape = "circle"
        if p2a_ratio > 4 * math.pi + ShapeDetector.P2A_RATIO_THRESHOLD:
            shape = "non-circle"
        return shape

    def detect_shape_obsolete(self, particle: Particle, img) -> str:
        x, y = particle.get_position()
        w, h = particle.get_bbox()

        #print(x, y)
        # 1. Cut image out
        # TODO: add a loop to gradually enlarge the bbox.
        
        # Coordinates of upper left corner of the crop
        x1 = x - ShapeDetector.BBOX_BUFFER if x - ShapeDetector.BBOX_BUFFER >= 0 else 0
        y1 = y - ShapeDetector.BBOX_BUFFER if y - ShapeDetector.BBOX_BUFFER >= 0 else 0

        # Coordinates of lower right corner of the crop
        x2 = x + w + ShapeDetector.BBOX_BUFFER \
            if x + w + ShapeDetector.BBOX_BUFFER <= img.shape[0] else img.shape[0]
        y2 = y + h + ShapeDetector.BBOX_BUFFER \
            if y + h + ShapeDetector.BBOX_BUFFER <= img.shape[1] else img.shape[1] 
        #print(x1, y1, x2, y2)
        cropped_img = img[y1:y2, x1:x2, :]  # First index is row, which is y in the x-y coordinate sense!
        cropped_img_binary = ShapeDetector.binary_threshold(cropped_img)
        
        # 2. Get contour
        cv.imwrite("./p_{:d}.jpg".format(particle.get_id()), cropped_img_binary)
        contour, bbox = ShapeDetector.detect_contour(cropped_img_binary)
        cv.drawContours(cropped_img, contour, -1, [0, 0, 255], 1)
        cv.imwrite("./p_{:d}_with_contour.jpg".format(particle.get_id()), cropped_img)
        
        # 3. Calculate ratio of contour length and area
        #if len(contours) > 1:
        #    print("Number of contours in the cropped image are: " + str(len(contours)))
        perimeter = cv.arcLength(contour, True)
        area = cv.contourArea(contour)
        p2a_ratio = perimeter ** 2 / area  # Take square to normalize
        print(p2a_ratio)
        shape = "circle"
        if p2a_ratio > 4 * math.pi + ShapeDetector.P2A_RATIO_THRESHOLD:
            shape = "non-circle"
        return shape

    def detect_contour(img):
        """
        Attribute:
            img Input cropped image in Opencv format with only one object.

        Return:
            contour of the particle in the given image crop.
            Array of [x, y, w, h] defining the bounding rectangulars.
        """
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        print(len(contours))
        contour = contours.pop(0)  # The first contour is the entire image. Remove.
        if len(contours) == 0:
            Logger.warning("Failed to find contour of a particle. " +
                           "The shape detection might be that of the bbox.")
        else:
            contour = contours.pop(0)

        x, y, w, h = cv.boundingRect(contour)
        return contour, [x, y, w, h]

    #TODO
    def detect_contours(img):
        """
        Attribute:
            img Input image in Opencv format.

        Return:
            contour
            Array of [x, y, w, h] defining the bounding rectangulars.
        """
        list_bbox = []
        #img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #cv.imwrite(add_suffix(output, "gray"), img_gray)
        #ret, img_threshold = cv.threshold(img_gray, threshold, 255, cv.THRESH_BINARY)
        #cv.imwrite(add_suffix(output, "threshold"), img_threshold)

        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        print(len(contours))
        contours.pop(0)  # The first contour is the entire image. Remove.
        contour = contours.pop(0)
        x, y, w, h = cv.boundingRect(contour)
        #n_removed = 0    # Number of unqualified particles being removed
        #for i in range(0, len(contours)):
        #    cnt = contours[i - n_removed]
        #    x, y, w, h = cv.boundingRect(cnt)
        #    if w * h <= ShapeDetector.AREA_THRESHOLD_LOWER or \
        #       w * h >= ShapeDetector.AREA_THRESHOLD_UPPER: 
        #        contours.pop(i - n_removed)
        #        n_removed += 1
        #        continue
        #    if draw_countour:
        #        img = cv.drawContours(img, contours, i - n_removed, color=[0, 0, 0])
        #    list_bbox.append([x, y, w, h])
        #return contours, list_bbox
        return contour, [x, y, w, h]

    def binary_threshold(img, threshold = 90, is_grayscale = False):
        if not is_grayscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, img_threshold = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
        return img_threshold

    def adaptive_threshold(img, method = cv.ADAPTIVE_THRESH_MEAN_C, blocksize = 15, offset = 0, 
                           is_grayscale = False):
        
        if not is_grayscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_threshold = cv.adaptiveThreshold(img, 255, method, cv.THRESH_BINARY, blocksize, offset)
        return img_threshold
    
    def crop_particle(particle, img, is_grayscale = False):
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

        #print(x, y)
        # 1. Cut image out
        # TODO: add a loop to gradually enlarge the bbox.
        
        # Coordinates of upper left corner of the crop
        x1 = x - ShapeDetector.BBOX_BUFFER if x - ShapeDetector.BBOX_BUFFER >= 0 else 0
        y1 = y - ShapeDetector.BBOX_BUFFER if y - ShapeDetector.BBOX_BUFFER >= 0 else 0

        # Coordinates of lower right corner of the crop
        x2 = x + w + ShapeDetector.BBOX_BUFFER \
            if x + w + ShapeDetector.BBOX_BUFFER <= img.shape[0] else img.shape[0]
        y2 = y + h + ShapeDetector.BBOX_BUFFER \
            if y + h + ShapeDetector.BBOX_BUFFER <= img.shape[1] else img.shape[1] 
        if is_grayscale:
            cropped_img = img[y1:y2, x1:x2]
        else:
            # First index is row, which is y in the x-y coordinate sense!
            cropped_img = img[y1:y2, x1:x2, :]
        return cropped_img