from typing import Sized, List

class Particle:
    """Particles recorded in combustion videos."""

    def __init__(self, id, time_frame, position: List[float], \
                 predicted_pos: List[float]=[0,0], bbox = [0, 0], bubble=False):
        self.id = id
        self.time_frame = time_frame
        self.position = position
        self.predict_pos = predicted_pos
        #self.x = self.position[0]
        #self.y = self.position[1]
        self.box = bbox  # length and width of identifying box around particle.
        self.size = bbox[0] * bbox[1]
        self.bubble = bubble    # boolean value for whether has bubble in particle

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
        return self.position

    def set_bbox(self, bbox):
        self.bbox = bbox
        self.size = self.bbox[0] * self.bbox[1]

    def get_bbox(self):
        return self.bbox
    
    def get_size(self):
        return self.size

    def set_bubble(self, bubble):
        self.bubble = bubble
    
    def has_bubble(self):
        return self.bubble

    def __str__(self) -> str:
        string = "Particle_id : {:4d}; Time_frame: {:4d}; ".format(self.id, self.time_frame) + \
                 "(x, y): ({:5.1f}, {:5.1f}); ".format(self.position[0], self.position[1]) + \
                 "Predicted (x, y): ({:5.1f}, {:5.1f}); ".format(self.predict_pos[0], 
                                                                   self.predict_pos[1]) + \
                 "Has_bubble: {:5s}".format(str(self.bubble))
        return string
    
    def __repr__(self) -> str:
        return "Particle(): particle identified in videos of combustion."
