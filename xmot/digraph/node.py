from __future__ import annotations
from typing import List, TYPE_CHECKING, Tuple
import sys

from xmot.digraph import commons

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
            position    : [int, int]   Centroid of positions of all particles
                                       between the start and end time.
    """

    def __init__(
            self,
            in_trajs: List[Trajectory],
            out_trajs: List[Trajectory],
            ptcl_ids: List[int] = None,
            type=None
        ):
        """

        """
        # ids of particles of all connected trajectories
        self.ptcl_ids = set(ptcl_ids) if ptcl_ids != None else set()
        self.in_trajs = in_trajs if in_trajs != None else []
        self.out_trajs = out_trajs if out_trajs != None else []

        # Post-processing properties
        self.type = type                        # a string describing the type of the event
                                                # "start", "end", "explosion", "collision"
        self.start_time = -1
        self.end_time = -1
        self.position = None

        # Deprecated
        # self.bbox_xy = None                     # upper-left and lower-right cornor of bbox

    def add_in_traj(self, traj: Trajectory):
        self.in_trajs.append(traj)
        self.ptcl_ids.add(traj.get_id())
        self.reset()

    def add_out_traj(self, traj: Trajectory):
        self.out_trajs.append(traj)
        self.ptcl_ids.add(traj.get_id())
        self.reset()

    def get_out_trajs(self):
        return self.out_trajs

    def is_out_traj(self, traj: Trajectory) -> bool:
        ids = [t.get_id() for t in self.out_trajs]
        return traj.get_id() in ids

    def get_in_trajs(self):
        return self.in_trajs

    def is_in_traj(self, traj: Trajectory) -> bool:
        ids = [t.get_id() for t in self.in_trajs]
        return traj.get_id() in ids

    def get_start_time(self):
        """
        Earliest time of all involved trajectories during the event, i.e. the earliest of end times
        of incoming trajctories and the earliest of start times of outgoing trajectories.
        Consider incoming trajectories first.

        Assume incoming trajectories have earliest end times earlier than earliest start times of
        outgoing trajectories.
        """
        # Test cases to consider: 1 in_traj, 2 in_traj, 1 out_traj, 2 out_traj, 2 in_traj and 2 out_traj.
        if self.start_time == -1:
            # Update start time from current set of trajs.
            _time = sys.maxsize
            for traj in self.in_trajs:
                # earliest end time of incoming trajectories.
                _time = traj.get_end_time() if traj.get_end_time() < _time else _time
            if _time == sys.maxsize:
                # self.in_trajs is empty. Consider the earliest time of start time of self.out_trajs
                for traj in self.out_trajs:
                    _time = traj.get_start_time() if traj.get_start_time() < _time else _time
            self.start_time = _time
        return self.start_time

    def get_end_time(self):
        """
        The latest time of all involved trajectories during the event, i.e. the latest of start times
        of outgoing trajectories and end times of incoming trajectories. Consider outgoing
        trajectories first.

        Assume largest start time of outgoing trajectories is larger than largest end time of
        incomning trajectories.
        """
        # Test cases to consider: 1 out_traj, 2 out_traj, 1 in_traj, 2 in_traj, 2 in_traj and 2 out_traj.
        if self.end_time == -1:
            # Update end time from current set of nodes.
            _time = -1
            for traj in self.out_trajs:
                # latest start time of all outgoing trajectories.
                _time = traj.get_start_time() if traj.get_start_time() > _time else _time
            if _time == -1:
                # self.out_trajs is empty. Consider the latest time of end time of self.in_trajs
                for traj in self.in_trajs:
                    _time = traj.get_end_time() if traj.get_end_time() > _time else _time
            self.end_time = _time
        return self.end_time

    def get_position(self):
        """
        Return size-weighted particle centroid position of all trajectories, including both
        incoming and outgoing ones.

        TODO: Use the weighted position at the event time (e.g. median of start/end times of trajs).
        """
        if self.position is None:
            total_size = 0
            total_x = total_y = .0

            _ptcls = []
            for traj in self.in_trajs:
                p = traj.get_end_particle()
                if p is not None:
                    _ptcls.append(p)
                p = traj.get_start_particle()
            for traj in self.out_trajs:
                if p is not None:
                    _ptcls.append(p)

            for p in _ptcls:
                total_size += p.get_size()
                total_x += p.get_size() * p.get_position()[0]
                total_y += p.get_size() * p.get_position()[1]
            if total_size != 0:
                self.position = [total_x / total_size, total_y / total_size]
        return self.position

    def get_bbox_node(self) -> List[int]:
        """
            Get the bbox coordinates in torch format, i.e. upper-left and lower-right corner.

            The bbox should be the rectangular enclosing all bboxes of end/start particles of
            incoming/outgoing trajectories. Note particles might not be at the same time frame.

            TODO: Perhaps we should use bboxes of particles through the start time until the end time.

            Return [x_min, y_min, x_max, y_max]
        """
        node_bbox = [
            commons.PIC_DIMENSION[0],   # upperleft_x. Maximum x value, picture width
            commons.PIC_DIMENSION[1],   # upperleft_y. Maximum y value, picture height
            0,                          # lowerright_x.
            0                           # lowerright_y.
        ]

        _bboxes = []
        for traj in self.in_trajs:
            _bboxes.append(traj.get_end_particle().get_contour_bbox_torch())

        for traj in self.out_trajs:
            _bboxes.append(traj.get_start_particle().get_contour_bbox_torch())

        for _bbox in _bboxes:
            node_bbox[0] = _bbox[0] if _bbox[0] < node_bbox[0] else node_bbox[0]
            node_bbox[1] = _bbox[1] if _bbox[1] < node_bbox[1] else node_bbox[1]
            node_bbox[2] = _bbox[2] if _bbox[2] > node_bbox[2] else node_bbox[2]
            node_bbox[3] = _bbox[3] if _bbox[3] > node_bbox[3] else node_bbox[3]

        return node_bbox

    def get_type(self):
        """
        Determine the type of event this node represents.
        """
        if self.type != None:
            return self.type

        # Analyze the number of trajectories in the in_trajs and out_trajs to determine type of event.
        # For each list, there're three situations: 0, 1, >1. In total, there are 3 x 3 = 9 cases.
        if len(self.in_trajs) == 0 and len(self.out_trajs) == 1:
            self.type = "start"
        elif len(self.in_trajs) == 1 and len(self.out_trajs) == 0:
            self.type = "end"
        elif len(self.in_trajs) <= 1 and len(self.out_trajs) > 1:
            # Case of len(in_trajs) = 0 could happen in explosions at the begining of a video.
            self.type = "micro-explosion"
        elif len(self.in_trajs) > 1 and len(self.out_trajs) > 1: # Subcategories
            if len(self.in_trajs) < len(self.out_trajs):
                self.type = "collision"
            elif len(self.in_trajs) == len(self.out_trajs):
                self.type = "crossing"
            else:
                self.type = "merge"
        elif len(self.in_trajs) > 1 and len(self.out_trajs) <= 1:
            # This could also be due to the flickering issue, i.e. same particle being detected
            # as different trajctories by Kalman filters.
            self.type = 'merge'
        elif len(self.in_trajs) == 1 and len(self.out_trajs) == 1:
            # This is often the same particle being detected as separate ones by Kalman filters.
            # Most likely due to the flickering phenomenon.
            self.type = "other same_ptcl"
        else:
            self.type = "other" # Both incoming and outgoing trajs are 0.
        return self.type

    def reset(self):
        """
            Reset post-processed properties of the node (like start_time, end_time, and type).

            Used when new trajectories are connected to the node.
        """
        self.start_time = -1
        self.end_time = -1
        self.position = None
        self.type = None

    # Use add_in_traj() and add_out_traj() to add particle id.
    #def add_particle(self, particle_id: int):
    #    if particle_id in self.ptcl_ids:
    #        Logger.debug("Particle-{:d} already registered for the node: {:s}".format(particle_id,
    #                                                                                  self))
    #        return
    #    self.ptcl_ids.append(particle_id)

    def __str__(self):
        string = "Node: Type: {:15s}; Incoming trajectories id: {:16s}; Outgoing trajectories id: {:16s}"
        string = string.format(
            self.get_type(),
            ",".join([str(traj.id) for traj in self.in_trajs]),
            ",".join([str(traj.id) for traj in self.out_trajs])
        )
        return string