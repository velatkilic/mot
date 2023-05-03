from typing import Sized, List
import cv2 as cv

class Particle:
    """Particles recorded in combustion videos.
    
    Attributes:
        position       : [int, int]     x, y positions in pixels of the upper left 
                                        corner of bbox. The x and y follows the OpenCV convention.
                                        With respect to the numpy.ndarray representation, 
                                        x is the column-index and y is the row-index.
        bbox           : [int, int]     Width (in x) and height (in y) in pixels of the bbox
    
    (Optional:)
        id             : int            ID of a particle
        time_frame     : int            Frame number of a particle in the video. It's used
                                        as time unit in the diagraph.
        predicted_pos  : [int, int]     Kalmen filter predicted x, y positions in pixels
                                        of the upper left corner of bbox
        contour        : numpy.ndarray  Contour object from OpenCV. The shape is always (n, 1, 2).
                                        "n" is the number of points in this contour.
        bubble         : Particle       Partible object representing the bubble. It only 
                                        needs position, bbox.
        shape          : str            Shape of particle. Permitted values are "circle", "non-circle".
        type           : str            Type of particle. "agglomerate", "shell", "particle".
                                        "shell": hollow shell; "particle": single solid particle.
        path_img       : str            Path to the source image.
    """

    def __init__(self, position: List[int], bbox: List[int], id = -1, time_frame = -1, \
                 predicted_pos: List[int] = [0,0], bubble=None, contour = None,
                 shape="N/A", type="N/A", path_img="N/A"):
        self.position = position
        self.bbox = bbox  # Width and height of bounding box.
        self.id = id
        self.time_frame = time_frame
        self.predict_pos = predicted_pos
        #self.x = self.position[0]
        #self.y = self.position[1]
        self.bubble = bubble
        self.contour = contour
        self.shape = shape
        self.type = type
        self.path_img = path_img

        # Derived values:
        self.bbox_area = bbox[0] * bbox[1]
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

    def set_position(self, position):
        self.position = position
    
    def get_position(self):
        """
        Return the upper left position of bbox.
        """
        return self.position # TODO: make deepcopy
    
    # Utility functions for opencv plotting.
    def get_top_left_position(self):
        return self.position
    
    def get_lower_right_position(self):
        return [self.position[0] + self.bbox[0], self.position[1] + self.bbox[1]]
    
    def get_top_left_position_reversed(self):
        return [self.position[1], self.position[0]]

    def get_center_position(self):
        """
        Return the position of the center of bbox.
        """
        return [self.position[0] + self.bbox[0] / 2, self.position[1] + self.bbox[1] / 2]

    def set_bbox(self, bbox):
        self.bbox = bbox
        self.area = self.bbox[0] * self.bbox[1]

    def get_bbox(self):
        return self.bbox
    
    def get_bbox_torch(self):
        """
        Return [x1, y1, x2, y2]
        """
        return [self.position[0], self.position[1], self.position[0] + self.bbox[0], self.position[1] + self.bbox[1]]
    
    def get_area_bbox(self):
        return self.bbox[0] * self.bbox[1]
    
    def get_area(self):
        return self.bbox[0] * self.bbox[1]

    def set_bubble(self, bubble):
        self.bubble = bubble
    
    def have_bubble(self):
        return self.bubble != None

    def get_contour_area(self, regenerate=False):
        if self.contour == None:
            return -1
        elif self.contour_contour_area == -1 or regenerate:
            self.contour_area = cv.contourArea(self.contour)
        return self.contour_area

    def get_contour_centroid(self, regenerate=False) -> List[int]:
        if self.contour == None:
            return None
        elif self.contour_centroid == None or regenerate:
            m1 = cv.moments(self.contour)
            self.contour_centroid = [int(m1["m10"]/m1["m00"]), int(m1["m01"]/m1["m00"])]
        return self.contour_centroid

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
            if self.bubble == None:
                return "{:s}_{:s}_{:s}".format(self.type, "no-bubble", self.shape)
            else:
                return "{:s}_{:s}_{:s}".format(self.type, "bubble", self.shape)
        elif self.type == "shell":
            return "{:s}_{:s}".format(self.type, self.shape)
        elif self.type == "agglomerate":
            return "agglomerate"
        
        return "N/A"

    def __str__(self) -> str:
        string = "Particle_id : {:4d}; Time_frame: {:4d}; ".format(self.id, self.time_frame) + \
                 "x, y: {:5.1f}, {:5.1f}; ".format(self.position[0], self.position[1]) + \
                 "bbox: {:5.1f}, {:5.1f}; ".format(self.bbox[0], self.bbox[1]) + \
                 "Area: {:6.2f}; ".format(self.get_area()) + \
                 "Type: {:12s}; ".format(self.type) + \
                 "Shape: {:12s}; ".format(self.shape) + \
                 "Has_bubble: {:5s}".format(str(self.bubble != None))
        return string
    
    def __repr__(self) -> str:
        return "Particle(): particle identified in 2D image"
