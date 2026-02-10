from typing import Sized, List
import cv2 as cv
import numpy as np
import copy
from xmot.utils.image_utils import get_contour_center
import math

class Particle:
    """Particles recorded in combustion videos.

    Attributes:
        position       : [int, int]     Cartesian coordinates of the particle centroid, predicted
                                        from the Kalman filter.
                                        The x and y follows the OpenCV convention. With respect to
                                        the numpy.ndarray representation of the image,
                                        x is the column-index and y is the row-index.
                                        This value is primarily used in velocity analysis, not for
                                        locating the particle, since it's Kalman-filter adjusted.
        bbox           : [int, int]     Width (in x) and height (in y) in pixels of the bbox,
                                        predicted from and adjusted by the Kalman filter.

    (Optional:)
        id             : int            ID of a particle
        time_frame     : int            Frame number of a particle in the video. It's used
                                        as time unit in the diagraph.
        contour        : numpy.ndarray  Contour from OpenCV. The shape is always (n, 1, 2).
                                        "n" is the number of points in this contour. This contour
                                        is the model-detected contour from the detection step. For
                                        shape detection and the exact location of the particle, use
                                        values derived from this contour. The 'position' and 'bbox'
                                        are Kalman filter adjusted.
        bubble         : Particle       Partible object representing the bubble. It only
                                        needs position, bbox.
        shape          : str            Shape of particle. Permitted values are "circle", "non-circle".
        type           : str            Type of particle. "agglomerate", "shell", "particle".
                                        "shell": hollow shell; "particle": single solid particle.
        path_img       : str            Path to the source image.

    (Deprecated:)
        predicted_pos  : [int, int]     Kalmen filter predicted x, y positions in pixels
                                        of the upper left corner of bbox
    """

    def __init__(self, position: List[int], bbox: List[int], id = -1, time_frame = -1, \
                 predicted_pos: List[int] = [0,0], bubble = None, contour: np.ndarray = None,
                 shape="N/A", type="N/A", path_img="N/A"):
        self.position = position
        self.bbox = bbox  # Width and height of bounding box.
        self.id = id
        self.time_frame = time_frame
        #self.predict_pos = predicted_pos
        #self.x = self.position[0]
        #self.y = self.position[1]
        self.bubble = bubble
        self.contour = np.array(contour).astype(np.int32) if contour is not None else None
        self.shape = shape
        self.type = type
        self.path_img = path_img

        # Derived values:
        # These values are derived once at initialization and not changed afterwards.
        if self.contour is not None and len(self.contour) != 0:
            _x, _y, _w, _h = cv.boundingRect(self.contour)
            self.cnt_bbox_area = _w * _h
            self.contour_area = round(cv.contourArea(self.contour))
            self.contour_centroid = get_contour_center(self.contour)
        else:
            # When contour is empty.
            self.cnt_bbox_area = -1
            self.contour_area = -1
            self.contour_centroid = None

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_time_frame(self, time_frame):
        self.time_frame = time_frame

    def get_time_frame(self):
        return self.time_frame

    def set_position(self, position) -> None:
        self.position = copy.deepcopy(position)

    def get_position(self) -> List[int]:
        """
        Return the Kalman-filter adjusted centroid position of the particle.
        """
        return copy.deepcopy(self.position)

    # Utility functions for opencv plotting.
    #def get_top_left_position(self):
    #    return self.position

    #def get_lower_right_position(self):
    #    return [self.position[0] + self.bbox[0], self.position[1] + self.bbox[1]]

    #def get_top_left_position_reversed(self):
    #    return [self.position[1], self.position[0]]

    #def get_center_position(self):
    #    """
    #    Return the position of the center of bbox.
    #    """
    #    return [self.position[0] + self.bbox[0] / 2, self.position[1] + self.bbox[1] / 2]

    def set_bbox(self, bbox):
        self.bbox = copy.deepcopy(bbox)

    def get_bbox(self) -> List[int]:
        return copy.deepcopy(self.bbox)

    def get_contour(self):
        # Check for array(None, dtype=object)
        return self.contour if self.contour is not None and self.contour.shape != () else None

    def get_contour_bbox_torch(self) -> List[int]:
        """
        Get the bbox in the torch convention, i.e. the coordinates of the upper left
        and lower right corner of the bounding box. Guarantee to return a non-Null value.

        Since this function is mostly used in benchmark and drawing bbox on images, it should
        always return a non-Null value. Therefore, if there exists a contour, the bounding box is
        the bounding rectangle of the contour. If there isn't a contour, this particle
        is most likely created by the Kalman filter for intermitent frames that a trajectory
        is not detected and we'll assume the Kalman-filter adjusted centroid is at the
        center of the Kalman-filter predicted bbox.

        Note usually the contour is from the object detection, and not adjusted by the Kalman filter.

        Return:
            [x1, y1, x2, y2]
        """
        if self.contour is None:
            x1 = round(self.position[0] - self.bbox[0] / 2)
            x2 = round(self.position[0] + self.bbox[0] / 2)
            y1 = round(self.position[1] - self.bbox[1] / 2)
            y2 = round(self.position[1] + self.bbox[1] / 2)
        else:
            x1, y1, _w, _h = cv.boundingRect(self.contour)
            x2 = x1 + _w
            y2 = y1 + _h

        # Always return a non-Null value.
        return [x1, y1, x2, y2]

    def get_area_bbox(self):
        """
        Get the bbox area. Note the bbox is the Kalman filter adjusted one.
        """
        return self.bbox[0] * self.bbox[1]

    def get_area_contour(self, regenerate=False) -> int:
        """
        Return the area of the contour. Return -1 when there is no contour.
        """
        if self.contour is None:
            return -1
        elif self.contour_area == -1 or regenerate:
            self.contour_area = round(cv.contourArea(self.contour))
        return self.contour_area

    def get_area_contour_bbox(self, regenerate=False) -> int:
        """
        Return the area of the enclosing bbox of the contour. Return -1 when there is no contour.
        """
        if self.contour is None:
            return -1
        elif self.cnt_bbox_area == -1 or regenerate:
            _x, _y, _w, _h = cv.boundingRect(self.contour)
            self.cnt_bbox_area = round(_w * _h)
        return self.cnt_bbox_area

    def get_position_contour(self, regenerate=False) -> List[int]:
        """
        Return the centroid position of the contour. This is the unadjusted position directly
        from the detection step. Return None when there is no contour.
        """
        if self.contour is None:
            return None
        elif self.contour_centroid is None or regenerate:
            self.contour_centroid = get_contour_center(self.contour)
        return copy.deepcopy(self.contour_centroid)

    #def get_area(self):
    #    return self.bbox[0] * self.bbox[1]

    def get_size(self) -> float:
        """
        This function should be used when trying to get the 'size' of the particle in analysis.
        It guarantees to return a positive size of the particle.

        Return the contour area when contour is not None. Otherwise, return the inscribed ellipse
        of the bbox predicted from Kalman-filter.

        Note: This function might return zero since we allow temporal disappearance of particle in
        a trajectory. Its width or height might be zero by Kalman filter adjustment.
        """
        if self.contour is not None:
            size = self.get_area_contour()
        else:
            # Area of inscribed ellipse of the bbox.
            # Note: bbox used here is most likely predicted by Kalman filters (since contour
            # is None) and has a zero value (since Kalman filter track the change of bbox size).
            size = math.pi * (self.bbox[0] / 2) * (self.bbox[1] / 2)
        return size

    def set_bubble(self, bubble):
        self.bubble = bubble  # For the particle objects, don't use deepcopy.

    def have_bubble(self):
        return self.bubble != None

    def set_shape(self, shape: str):
        self.shape = shape

    def get_shape(self):
        return self.shape

    def set_type(self, type):
        self.type = type

    def get_type(self):
        return self.type

    def get_label(self):
        label = self.type
        if self.type == "particle":
            if self.bubble is None:
                return "{:s}_{:s}_{:s}".format(self.type, "no-bubble", self.shape)
            else:
                return "{:s}_{:s}_{:s}".format(self.type, "bubble", self.shape)
        elif self.type == "shell":
            return "{:s}_{:s}".format(self.type, self.shape)
        elif self.type == "agglomerate":
            return "agglomerate"

        return "N/A"

    def short_rep(self) -> str:
        return f"ID: {self.id:4d}; time: {self.time_frame:4d}; position: " + "{:4d}, {:4d}".format(*self.position)

    def __str__(self) -> str:
        position_cnt = self.get_position_contour()
        if self.get_position_contour() is None:
            position_cnt = self.position

        # When there's no contour, use the Kalman filter predicted values as contour position
        # and contour area. Although Kalman filter only predicts bbox area, we use it to approximate
        # the contour area.

        # Particle size is equivalent to contour area when a contour exists.
        string = "Particle_id : {:4d}; Time_frame: {:4d}; ".format(self.id, self.time_frame) + \
                 "x, y: {:6d}, {:6d}; ".format(*self.position) + \
                 "bbox (w, h): {:6d}, {:6d}; ".format(*self.bbox) + \
                 "Particle size: {:6f}; ".format(self.get_size()) + \
                 "Contour centroid: {:6d}, {:6d}; ".format(*position_cnt) + \
                 "Type: {:12s}; ".format(self.type) + \
                 "Shape: {:12s}; ".format(self.shape) + \
                 ""
                 #"Has_bubble: {:5s}".format(str(self.bubble != None))
                 # Can't detect bubble for now. No point of printing it.
        return string

    def __repr__(self) -> str:
        return "Particle (x, y, w, h): {:.2d} {:.2d} {:.2d} {:.2d}".format(*self.position, *self.bbox)
