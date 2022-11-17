from typing import Sized, List

class Particle:
    """Particles recorded in combustion videos.
    
    Attributes:
        position       : [int, int] x, y positions in pixels of the upper left 
                                    corner of bbox
        bbox           : [int, int] Width (in x) and height (in y) in pixels of the bbox
    (Optional:)
        id             : int        ID of a particle
        time_frame     : int        Frame number of a particle in the video. It's used
                                    as time unit in the diagraph.
        predicted_pos  : [int, int] Kalmen filter predicted x, y positions in pixels
                                    of the upper left corner of bbox
        bubble         : Particle   Partible object representing the bubble. It only 
                                    needs position, bbox.
        shape          : str        Shape of particle. Permitted values are "circle", "non-circle".
        type           : str        Type of particle. "agglomerate", "shell", "particle".
                                    "shell": hollow shell; "particle": single solid particle.
        path_img       : str        Path to the source image.
    """

    def __init__(self, position: List[int], bbox: List[int], id = -1, time_frame = -1, \
                 predicted_pos: List[int] = [0,0], bubble=None, shape="", type="", path_img=""):
        self.position = position
        self.bbox = bbox  # Width and height of bounding box.
        self.id = id
        self.time_frame = time_frame
        self.predict_pos = predicted_pos
        #self.x = self.position[0]
        #self.y = self.position[1]
        self.area = bbox[0] * bbox[1]
        self.bubble = bubble
        self.shape = shape
        self.type = type
        self.path_img = path_img

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
        return self.area

    def set_bubble(self, bubble):
        self.bubble = bubble
    
    def have_bubble(self):
        return self.bubble != None

    def set_shape(self, shape: str):
        self.shape = shape

    def get_shape(self):
        return self.shape
    
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
        
        return ""

    def __str__(self) -> str:
        string = "Particle_id : {:4d}; Time_frame: {:4d}; ".format(self.id, self.time_frame) + \
                 "x, y: {:5.1f}, {:5.1f}; ".format(self.position[0], self.position[1]) + \
                 "Area: {:6.2f}; ".format(self.get_area()) + \
                 "Shape: {:12s}; ".format(self.shape) + \
                 "Has_bubble: {:5s}".format(str(self.bubble != None))
        return string
    
    def __repr__(self) -> str:
        return "Particle(): particle identified in videos of combustion."
