from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np
import copy
import digraph.commons as commons
from logger import Logger
from digraph.utils import ptcl_distance

if TYPE_CHECKING:
    from node import Node
    from particle import Particle

class Trajectory:
    """
        Edge of directed graph. Each trajectory is composed of a collection
        of particle objects representing the same physical particle at different
        time frame.

        Attributes:
            id: particle id of the underlying particle.
            ptcls: Particle of same ID at different time frames.
            kalmanfilter: The kalmanfilter that can predict position of underlying particle
                after the end_time.
            start_node: Node marking the start of the trajectory. It could be that the particle
                rises from an event, it for first time enter the image, or it exists at the 
                beginning of the video.
            end_node: Node marking the end of the trajectory. It could be that the particle 
                leaves the image, the video ends, or an event happenes.
            start_time: The time frame the underlying particle is firstly detected.
            end_time: The time frame the underlying paritcle is lastly detected.
            velocity: The average velocity averaged over all time frames.
        
        Notes：
        1. self.ptcls should be sorted in ascending order of time frames. The last one in the list 
        is the one with highest value of time frame.
        2. self.id is the only necessary argument for the constructor. All the other attributes
        can be added afterwards.

    """

    def __init__(self, id, ptcls: List[Particle], kalmanfilter= None,
                 start_node: Node = None, end_node: Node = None):
        """

        """
        self.id = id
        self.ptcls = ptcls
        self.kalmanfilter = kalmanfilter
        self.start_node = start_node
        self.end_node = end_node
        
        # Post-processed properties. They should not be set in constructor, but be extracted 
        # after the trajectory is fully built.
        self.start_time = 0 
        self.end_time = 0
        self.velocity = float("nan")

        # Status flag.
        self.sorted = False

    def add_particles(self, particles: List[Particle], merge: bool = False) -> bool:
        ptcls_backup = copy.deepcopy(self.ptcls)
        for p in particles:
            if not self.add_particle(p, merge):
                self.ptcls = ptcls_backup
                return False
        return True

    def add_particle(self, particle: Particle, merge: bool = False) -> bool:
        """
            Add particle to the trajectory.

            Args:
                particle: particle to be added. Don't add this particle if its id
                    is different from the trajectory id, unless merge is True.
                merge: whether force the adding of the particle. If true, change
                    the id of the particle to the id of this trajectory. However,
                    it the particle of the base id already exists at this time,
                    don't merge and return False.
        """
        if self.id != particle.id:
            if not merge:
                Logger.warning("Cannot add particle! New particle has different "
                               "id as the existing particles: " +
                               "{:d} {:d}".format(particle.id, self.id))
                return False
            elif self.has_particle(particle.time_frame):
                Logger.warning("Cannot add particle! Particle already exists at this time frame: " +
                               "{:d} {:d} {:d}".format(particle.time_frame, self.id, particle.id))
                return False
            else:
                Logger.detail(
                    "Merge new particle of different id to this trajectory: " +
                    "{:3d} {:3d}".format(particle.id, self.id))
                particle.set_id(self.id)
        
        self.ptcls.append(particle)
        self.reset()
        return True

    def set_start_node(self, node):
        self.start_node = node
    
    def set_end_node(self, node):
        self.end_node = node
    
    def sort_particles(self):
        """
            Sort self.ptcls in ascending order of time frames.

            Enforce a re-sort even if the list has been sorted.
        """
        if len(self.ptcls) > 0:
            list.sort(self.ptcls, key=lambda p: p.get_time_frame()) # in-place sort
            self.start_time = self.ptcls[0].time_frame
            self.end_time = self.ptcls[-1].time_frame
        else:
            Logger.warning("Trajectory is empty.")
            self.start_time = 0
            self.end_time = 0
        self.sorted = True

    # Post-processing functions
    def is_sorted(self):
        return self.sorted

    def get_start_time(self) -> int:
        if not self.is_sorted():
            self.sort_particles()
        return self.start_time

    def get_end_time(self) -> int:
        """
            The last time frame that the underlying particle exists in the video. Inclusive.
        """
        if not self.is_sorted():
            self.sort_particles()
        return self.end_time

    def get_life_time(self) -> int:
        return self.get_end_time() - self.get_start_time() + 1
    
    def get_start_position(self) -> List[float]:
        if not self.is_sorted:
            self.sort_particles()
        if len(self.ptcls) > 0:
            return self.ptcls[0].position
        else:
            return None

    def get_end_position(self) -> List[float]:
        if not self.is_sorted:
            self.sort_particles()
        if len(self.ptcls) > 0:
            return self.ptcls[-1].position
        else:
            return None
    
    def has_particle(self, time: int) -> bool:
        """
            Whether a particle already exists in this trajectory at the given time.
        """
        for p in self.ptcls:
            if p.time_frame == time:
                return True
        return False

    def get_snapshots(self, start: int, end: int) -> List[Particle]:
        """Retrun a deep copy of lists of particles between the time interval (inclusive)."""
        snapshot = []
        for p in self.ptcls:
            if start <= p.time_frame and p.time_frame <= end:
                snapshot.append(copy.deepcopy(p))
        return snapshot

    def get_particle(self, time: int) -> Particle:
        self.sort_particles()
        if time < self.start_time or time > self.end_time:
            Logger.error("Specified time is outside the life time of the trajectory: " + \
                         "{:3d} {:4d} {:4d} {:4d}".format(self.id, time, 
                                                          self.start_time, self.end_time))
            return None
        for p in self.ptcls:
            if p.time_frame == time:
                return p
        Logger.detail("No particle exists in this trajectory at the time: " + 
                      "{:3d} {:4d}".format(self.id, time))
        return None
    
    def get_particles(self) -> List[Particle]:
        return self.ptcls

    def get_velocity(self) -> float:
        """
            Calculate and return the average velocity averaged over velocities at all time 
            frames of the underlying particle.
        """
        if not self.sorted:
            self.sort_particles()
        velocities = []
        for i in range(1, len(self.ptcls)):
            p1 = self.ptcls[i - 1]
            p2 = self.ptcls[i]
            if p1.time_frame == p2.time_frame:
                # Multiple particles exists for the same time frame.
                Logger.error("Something is wrong this trajectory: {:d}".format(self.id))
                continue
            velocities.append(ptcl_distance(p1, p2) / (p2.time_frame - p1.time_frame))
        if len(velocities) > 0:
            self.velocity = np.average(velocities)
        return self.velocity

    def predict_next_location(self) -> List[float]:
        """
            Return a pair of x, y values as the predicted position at the next time frame.
        """
        last_position = self.ptcls[-1].positions
        new_position = np.add(last_position, self.velocity * commons.TIMEFRAMELENGTH)
        return new_position

    def reset(self):
        """
            Reset post-processed properties of the trajectory when new particles are
            appended to the trajectory.
        """
        self.start_time = 0
        self.end_time = 0
        self.velocity = 0.0
        self.sorted = False

    def __str__(self) -> str:
        string = "Trajectory: Particle id: {:3d}; ".format(self.id) + \
                 "Start time: {:4d}; ".format(self.get_start_time()) + \
                 "End time: {:4d}; ".format(self.get_end_time()) + \
                 "Average velocity: {:5.2f}".format(self.get_velocity())
        return string