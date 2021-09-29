from __future__ import annotations
from typing import List, TYPE_CHECKING, Tuple
from logger import Logger
from digraph.commons import PIC_DIMENSION
import sys

if TYPE_CHECKING:
    from trajectory import Trajectory
    from particle import Particle

class Node:
    """
        Node type of directed graph, representing events of particles, like
        micro-explosion, collision, start and end of a eventless trajectory, and etc.

        Attributes:
            ptcl_ids    : [int]        List of particle (trajectories) ids involved 
                                       in this event.
            in_trajs    : [Trajectory] Incoming trajectories.
            out_trajs   : [Trajectory] Outgoing trajectories.
            type        : str          A word denoting type of the event: 
                                       "start", "end", "collision", "explosion"
            start_time  : int          Time frame marking the start of the event.
            end_time    : int          Time frame marking the end of the event.
            position    : [int, int]   (?) Centroid of positions of all particles 
                                       between the start and end time.
    """

    def __init__(self, ptcl_ids: List[int] = None, in_trajs: List[Trajectory] = None, 
                 out_trajs: List[Trajectory] = None, type=None):
        """
            
        """
        # ? ids of particles of all connected trajectories
        self.ptcl_ids = ptcl_ids if ptcl_ids != None else []
        self.in_trajs = in_trajs if in_trajs != None else []
        self.out_trajs = out_trajs if out_trajs != None else []

        # Post-processing properties
        self.type = type                        # a string describing the type of the event
                                                # "start", "end", "explosion", "collision"
        self.start_time = -1
        self.end_time = -1
        #self.position = [0.0, 0.0]
        self.bbox_xy = None                     # upper-left and lower-right cornor of bbox

    #def get_position(self):
    #    return self.position

    def get_start_time(self):
        if self.start_time == -1:
            # Update start time from current set of nodes.
            _time = sys.maxsize
            for traj in self.in_trajs:
                # earliest end time of incoming trajectories.
                _time = traj.get_end_time() if _time > traj.get_end_time() else _time
            if _time == sys.maxsize:
                # self.in_trajs is empty. Consider the earliest time of start time of self.out_trajs
                for traj in self.out_trajs:
                    _time = traj.get_start_time() if _time > traj.get_start_time() else _time
            self.start_time = _time
        return self.start_time
    
    def get_end_time(self):
        if self.end_time == -1:
            # Update end time from current set of nodes.
            _time = -1
            for traj in self.out_trajs:
                # latest start time of all outgoing trajectories.
                _time = traj.get_start_time() if _time < traj.get_start_time() else _time
            if _time == -1:
                # self.out_trajs is empty. Consider the latest time of end time of self.in_trajs
                for traj in self.in_trajs:
                    _time = traj.get_end_time() if _time < traj.get_end_time() else _time
            self.end_time = _time
        return self.end_time
    
    def get_bbox(self) -> List[Tuple[int]]: 
        """
            Get the boundary coordinates of all starting and end positions of trajectories.
            
            Don't consider the change of bbox as time goes by. For example, in explosion, new
            trajectories will emerge at a later time, and inevitable increase the bbox.
        """
        _upperleft_x = 624
        _upperleft_y = 640
        _lowerright_x = 0
        _lowerright_y = 0
        for traj in self.in_trajs:
            pos = traj.get_end_position()
            _upperleft_x = pos[0] if _upperleft_x > pos[0] else _upperleft_x
            _upperleft_y = pos[1] if _upperleft_y > pos[1] else _upperleft_y
            _lowerright_x = pos[0] if _lowerright_x < pos[0] else _lowerright_x
            _lowerright_y = pos[1] if _lowerright_y < pos[1] else _lowerright_y
        for traj in self.out_trajs:
            pos = traj.get_start_position()
            _upperleft_x = pos[0] if _upperleft_x > pos[0] else _upperleft_x
            _upperleft_y = pos[1] if _upperleft_y > pos[1] else _upperleft_y
            _lowerright_x = pos[0] if _lowerright_x < pos[0] else _lowerright_x
            _lowerright_y = pos[1] if _lowerright_y < pos[1] else _lowerright_y
        
        # <todo>: Temporary. For node having only one trajectory, give them a default 20 pixel box 
        # size.
        _default_box_size = 20
        if self.get_type() == "end": _default_box_size = 15
        if _upperleft_x == _lowerright_x:
            _upperleft_x -= _default_box_size / 2
            _lowerright_x += _default_box_size / 2
        if _upperleft_y == _lowerright_y:
            _upperleft_y -= _default_box_size / 2
            _lowerright_y += _default_box_size / 2
        return [(_upperleft_x, _upperleft_y), (_lowerright_x, _lowerright_y)]

    def get_type(self):
        if self.type != None:
            return self.type
        
        # Analysis in_trajs and out_trajs to determine self.type
        if len(self.in_trajs) == 0 and len(self.out_trajs) == 1:
            self.type = "start"
        elif len(self.in_trajs) == 1 and len(self.out_trajs) == 0:
            self.type = "end"
        elif len(self.in_trajs) == 1 and len(self.out_trajs) > 1:
            self.type = "explosion"
        elif len(self.in_trajs) > 1 and len(self.out_trajs) > 1:
            self.type = "collision"
            # check whether it is crossing instead of collision.
            # <todo>
        return self.type

    def add_in_traj(self, traj):
        self.in_trajs.append(traj)
        if traj.id not in self.ptcl_ids:
            self.ptcl_ids.append(traj.id)
        self.reset()
    
    def add_out_traj(self, traj):
        self.out_trajs.append(traj)
        if traj.id not in self.ptcl_ids:
            self.ptcl_ids.append(traj.id)
        self.reset()

    def reset(self):
        """
            Reset post-processed properties of the node. Used when new trajectories are connected
            to the node.
        """
        self.start_time = -1
        self.end_time = -1
        self.type = None
        self.bbox_xy = None

    # Use add_in_traj() and add_out_traj() to add particle id.
    #def add_particle(self, particle_id: int):
    #    if particle_id in self.ptcl_ids:
    #        Logger.debug("Particle-{:d} already registered for the node: {:s}".format(particle_id, 
    #                                                                                  self))
    #        return
    #    self.ptcl_ids.append(particle_id)

    def __str__(self):
        """
            Print an identifier of the node.
            <todo>
        """
        string = "Node: Incoming trajectories id: "
        for traj in self.in_trajs:
            string += str(traj.id) + ","
        string = string.strip(",") + "; "
        string += "Outgoing trajectories id: "
        for traj in self.out_trajs:
            string += str(traj.id) + ","
        string = string.strip(",")
        return string