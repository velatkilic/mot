from __future__ import annotations
from typing import List, TYPE_CHECKING, Union
import numpy as np
import math
import copy
from collections import Counter

import xmot.digraph.commons as commons
from xmot.mot.kalman import MOT
from xmot.logger import Logger
from xmot.digraph.utils import ptcl_distance, BACK_TRACE_LIMIT, vector_angle

if TYPE_CHECKING:
    from node import Node
    from particle import Particle

class Trajectory:
    """
        Edge of directed graph. Each trajectory is composed of a collection
        of particle objects representing the same physical particle at different
        time frame.

        Attributes:
            id            : particle id of the underlying particle.
            ptcls         : Particle of same ID at different time frames.
            kalmanfilter  : The kalmanfilter that can predict position of underlying particle
                            after the end_time.
            start_node    : Node marking the start of the trajectory. It could be that the particle
                            emerges from an event, it just enters the video, or this node is the
                            beginning of the video.
            end_node      : Node marking the end of the trajectory. It could be that the particle
                            leaves the image, the video ends, or an event happenes.
            start_time    : The time frame the underlying particle is firstly detected.
            end_time      : The time frame the underlying paritcle is lastly detected.
            velocity      : The average velocity averaged over all time frames.

        Notes:
        1. self.ptcls should be sorted in ascending order of time_frame. The last one in the list
        is the one exists later in the video.
        2. self.id is the only necessary argument for the constructor. All the other attributes
        can be added afterwards.

    """

    def __init__(self, id, ptcls: List[Particle], start_node: Node = None,
                 end_node: Node = None, kalmanfilter = None):
        """

        """
        self.id: int = id
        self.ptcls: List[Particle] = ptcls
        self.start_node: Node = start_node
        self.end_node: Node = end_node

        self.kalmanfilter = kalmanfilter # Deprecated

        # helper variables
        self.ptcl_time_map = {p.get_time_frame(): p for p in ptcls} if ptcls is not None else {}

        # Post-processed properties. They should not be set in constructor, but be extracted
        # after the trajectory is fully built.
        self.start_time = -1
        self.end_time = -1
        self.avg_velocity = float("nan")    # Average velocity over the lifetime of the trajectory.
        self.velocity_vectors = None

        # Status flag to control internal behaviour.
        self.__sorted = False           # whether particles at different time frame of this trajectory
                                        # are sorted in ascending order of time_frame. Properties
                                        # rely on sequential processing of the list of particles
                                        # will be wrong if particles are not in chronological order.
                                        # Sorting is requried before calculation of any of these
                                        # properties.

    @staticmethod
    def break_trajectory(traj: Trajectory, draw: bool = False) -> Union[Trajectory, None]:
        """
        Inspecting the particles of the trajectory and see whether the trajectory consists of
        different particles. If so, break them into separate ones.

        Kalman filters could group different particles in one trajectory due to the way cost_matrix
        is defined. The size of the particle is emphasized as much as the position. Particles
        of vastly different size but at close region could be grouped into one trajectory. This
        leads to micro-explosion lacking parent trajectory since the parent particle is grouped
        with one of the children trajectory. This function mainly checks the particle size (
        and velocity vectors). When there are abrupt and major changes in size and velocity vector,
        we consider the given trajectory should be break into two distinct ones.

        Args:
            traj:       Trajectory to inspect
            draw_dir:   Folder to draw the original and breakup trajectories into.

        Return:
            Trajectory | None: The new trajectory separated from the given trajectory
        """
        separate = False
        traj._compute_velocity_vectors()

        particles: List[Particle] = traj.get_particles()
        change_ratios = []
        change_values = []
        valid_indices = []
        for i in range(0, len(particles) - 1):
            size_i = particles[i].get_size()
            if size_i == 0:  # Avoid divide by zero.
                continue
            valid_indices.append(i)

            size_j = particles[i+1].get_size()
            change_ratios.append(abs((size_i - size_j) / size_i))
            change_values.append(abs(size_i - size_j))

        indices = [
            i
            for i, ratio, value in zip(valid_indices, change_ratios, change_values)
            if ratio > 0.5 and value > 50
        ]

        # time_frames[0] will be the time point to break the trajectory
        time_frames = [particles[i+1].get_time_frame() for i in indices]

        # Criteria:
        # - Change over 50% and with an absolute change above 50 pixel**2.
        # - Have one and only one of such case.
        # - If breakup, both the old and new trajectory will have significant life time (more than 3 frames)
        if (
            len(indices) == 1 and
            traj.get_end_time() - time_frames[0] > 3 and
            time_frames[0] - traj.get_start_time() > 3 and
            traj.get_particle_by_time(time_frames[0]) is not None
            #not traj.get_particle_by_time(time_frames[0]).close_to_edge()
        ):
            separate = True

        # TODO: Should we have another criterion for velocity vectors?

        if not separate:
            return None

        # From break_time onwards, particles belong to the new trajectory.
        break_time = particles[indices[0] + 1].get_time_frame()
        Logger.debug(f"Breaking up trajectory-{traj.get_id()} at {break_time}")

        # Note: shallow copy. We deliberately keep the same object references.
        new_traj = Trajectory(id = -1, ptcls=particles[indices[0] + 1:])

        traj.reset() # Mark as unsorted since the particle list change
        for p in new_traj.get_particles():
            traj.remove_particle(p)

        return new_traj


    def merge_particles(self, particles: List[Particle]) -> bool:
        """
        Merge a list of particles into this trajectory. If ids are different to the id of
        this trajectory, change ids of the particles to that of this trajectory.
        Report error if there're already particles in this trajectory at covering
        time frames or particle ID doesn't match trajectory id.

        Note: the list of particles should also be objects in the digraph, so that changes
        of particle ids (if merged) are reflected back in the digraph particle list.
        """
        # ptcls_backup = copy.deepcopy(self.ptcls) # TODO: improve memory efficiency

        # Perform a dry run to check whether the list of new particles can be added to this
        # trajectory. Perform the adding only after the list is valid. This allows an "atomic"
        # modification of the trajectory.
        # legit = True
        for p in particles:
            #if p.id != self.id:
            #    if not merge:
            #        Logger.error("Cannot add particle! New particle has different "
            #                    f"id as the existing particles: {p.id} {self.id}")
            #        legit = False
            #    elif self.has_particle(p.time_frame):
            if self.has_particle(p.time_frame):
                Logger.error("Cannot merge particle! A particle already exists "
                            f"at this time frame: {p.get_time_frame()} {self.id} {p.id}")
                return False
        #if not legit:
        #    return False

        for p in particles:
            if p.id != self.id:
                Logger.detail("Force merging a new particle of different id to this trajectory: " +
                             f"{p.id} {self.id}")
                p.set_id(self.id)
            #if not self.add_particle(p, merge):
            #    #Logger.error("Fail to add particle {:d} into trajectory {:d}".format(
            #    #    p.get_id(), self.id))
            #    self.ptcls = ptcls_backup
            #    return False
            self.ptcls.append(p)
            self.ptcl_time_map[p.get_time_frame()] = p
        if self.__sorted:
            self.reset()
        return True

    #def add_particle(self, particle: Particle, merge: bool = False) -> bool:
    #    """
    #        [Obsolete]
    #        Add particle to the trajectory, by appending this particle to the internal
    #        list of belonging particles and update the list of time frames this trajectory
    #        spans.
    #
    #        Args:
    #            particle: particle to be added. Don't add this particle if its id
    #                is different from the trajectory id, unless merge is True.
    #            merge: whether force the adding of the particle. If true, change
    #                the id of the particle to the id of this trajectory. Otherwise,
    #                don't merge and return False.
    #    """
    #    if self.id != particle.id:
    #        if not merge:
    #            Logger.error("Cannot add particle! New particle has different "
    #                         "id as the existing particles: " +
    #                         "{:d} {:d}".format(particle.id, self.id))
    #            return False
    #        elif self.has_particle(particle.time_frame):
    #            Logger.error("Cannot merge particle! A particle already exists at this time frame: " +
    #                         "{:d} {:d} {:d}".format(particle.time_frame, self.id, particle.id))
    #            return False
    #        else:
    #            Logger.detail("Force merging a new particle of different id to this trajectory: " +
    #                          "{:3d} {:3d}".format(particle.id, self.id))
    #            particle.set_id(self.id)
    #
    #    self.ptcls.append(particle)
    #    self.ptcl_time_map[particle.get_time_frame()] = particle
    #    if self.__sorted:
    #        self.reset()
    #    return True

    def append_particle(self, particle: Particle) -> None:
        """
        Simple append the particle to the particle list without any checks.
        """
        self.ptcls.append(particle)
        self.ptcl_time_map[particle.get_time_frame()] = particle

        # There's no guarantee that the new particle is later in time than existing particles.
        # Have to re-sort in chronological order.
        if self.__sorted:
            self.reset()

    def remove_particle(self, particle: Particle) -> Particle:
        """Remove the given particle from the trajectory. Match particle id and time."""
        if self.__sorted:
            self.reset()

        # Don't need to worry about index shifting since we only remove one item.
        for i, p in enumerate(self.ptcls):
            if particle.get_id() == p.get_id() and particle.get_time_frame() == p.get_time_frame():
                self.ptcls.pop(i)
                del self.ptcl_time_map[p.get_time_frame()]
                break

        return p

    def remove_no_contour_particles_at_end(self) -> List[Particle]:
        """
        The Kalman filter allows particles to remain undetected for a few frames before
        removing them entirely. This results in two fictitious particles without
        contours at the end of each Kalman filter trajectory. Check for their
        existence and remove them.
        """
        if not self.__sorted:
            self.sort_particles()

        count = 0
        removed_ptcls = []
        for ptcl in self.ptcls[::-1]: # Reverse iteration.
            if ptcl.contour is None:
                self.ptcls.remove(ptcl)
                del self.ptcl_time_map[ptcl.get_time_frame()]
                count += 1
                removed_ptcls.append(ptcl)
                continue

            break # Stop at first particle with contour.

        if count > 0:
            self.reset()

        if count > MOT.UNDETECTION_THRESHOLD:
            Logger.warning(
                f"Removed {count} frames from the end of trajectory-{self.id}!" \
                F" Should not be larger than {MOT.UNDETECTION_THRESHOLD}."
            )

        return removed_ptcls

    def set_start_node(self, node):
        self.start_node = node

    def set_end_node(self, node):
        self.end_node = node

    def sort_particles(self):
        """
            Sort self.ptcls in ascending order of time frames. (-1 is the last frame)

            Enforce a re-sort even if the list has been sorted.
        """
        self.reset()
        if self.ptcls != None and len(self.ptcls) > 0:
            list.sort(self.ptcls, key=lambda p: p.get_time_frame()) # in-place sort
            self.start_time = self.ptcls[0].time_frame
            self.end_time = self.ptcls[-1].time_frame
        else:
            Logger.debug("Sorted an empty trajectory {:d}".format(self.id))

        self.__sorted = True

    # Post-processing functions
    def get_id(self) -> int:
        return self.id

    def set_id(self, id: int, overwrite_particle_id = True) -> None:
        self.id = id

        # Align particle id with trajectory id.
        if overwrite_particle_id:
            for p in self.ptcls:
                p.set_id(id = self.id)

    def get_start_node(self):
        return self.start_node

    def get_end_node(self):
        return self.end_node

    def get_start_time(self) -> int:
        if not self.__sorted:
            self.sort_particles()
        return self.start_time

    def get_end_time(self) -> int:
        """
            The last time frame that the underlying particle exists in the video. Inclusive.
        """
        if not self.__sorted:
            self.sort_particles()
        return self.end_time

    def get_life_time(self) -> int:
        """
        Return the number of frames in which the trajectory exists.
        """
        return self.get_end_time() - self.get_start_time() + 1

    # TODO: Use get_position_by_time to simplify code.
    def get_position_start(self) -> List[float]:
        """
        Return:
            [float, float] Position of the underlying particle at the start of the trajectory.
                           Kalman-filter adjusted.
        """
        if not self.__sorted:
            self.sort_particles()
        if len(self.ptcls) > 0:
            return self.ptcls[0].get_position()
        else:
            return None

    def get_position_end(self) -> List[float]:
        """
        This is equivalent to traj.get_end_particle().get_position()

        Return:
            [float, float] Position of the underlying particle at the end of the trajectory.
                           Kalman-filter adjusted.
        """
        if not self.__sorted:
            self.sort_particles()
        if len(self.ptcls) > 0:
            return self.ptcls[-1].get_position()
        else:
            return None

    def get_position_by_time(self, time: int) -> List[float]:
        """
        Get the centroid position of underlying particle at a specified time-frame.

        Attribute:
            time    int     Desired time frame.

        Return:
            [float, float]  Position of particle at frame "time". None if not exist at the time.
        """
        if time in self.ptcl_time_map.keys():
            return self.ptcl_time_map[time].get_position()
        elif len(self.ptcls) < 2:
            Logger.detail("Cannot backtrace or predict position for trajectory existing for " \
                          "less than two frames.")
            return None

        if not self.__sorted:
            self.sort_particles()

        if time < self.get_start_time() and self.get_start_time() - time <= BACK_TRACE_LIMIT:
            # 'time' is before (smaller) the start of the trajectory
            # Back trace to estimate positions in past time (no more than 3 frames).
            # Use instant velocity at the first two frames to backtrace positions
            # <TODO> We should use the average of as many as possible instant velocities that don't
            # drastically change direction. (angle between velocity vectors). Even better, the
            # Kalman filter.
            Logger.debug("Backtrace position of trajectory-{:d} at frame-{:d}.".format(self.id, time))

            p1 = self.ptcls[0].get_position()
            p2 = self.ptcls[1].get_position() # Later in time
            delta_t = self.ptcls[1].get_time_frame() - self.ptcls[0].get_time_frame() # Positive
            if delta_t == 0:
                Logger.debug(f"The first two particles of trajectory-{self.id} have equal time frame! Cannot backtrace.")
                return p1 # The first particle
            return_position = [0, 0]
            return_position[0] = p1[0] - (p2[0] - p1[0]) / delta_t * (self.ptcls[0].get_time_frame() - time)
            return_position[1] = p1[1] - (p2[1] - p1[1]) / delta_t * (self.ptcls[0].get_time_frame() - time)
            return return_position
        elif time > self.get_end_time() and time - self.get_end_time() <= BACK_TRACE_LIMIT:
            # 'time' is after the end of the trajectory
            # Forward trace to predict positions of particles in a future time.
            Logger.debug("Predict position of trajectory-{:d} at frame-{:d}.".format(self.id, time))
            # Use instant velocity at the last two frames to predict positions

            p1 = self.ptcls[-1].get_position()
            p2 = self.ptcls[-2].get_position() # Second to the last frame
            delta_t = self.ptcls[-1].get_time_frame() - self.ptcls[-2].get_time_frame() # Positive
            if delta_t == 0:
                Logger.debug(f"The last two particles of trajectory-{self.id} have equal time frame! Cannot predict.")
                return p1 # The last particle
            return_position = [0, 0]
            return_position[0] = p1[0] + (p1[0] - p2[0]) / delta_t * (time - self.ptcls[-1].get_time_frame())
            return_position[1] = p1[1] + (p1[1] - p2[1]) / delta_t * (time - self.ptcls[-1].get_time_frame())
            return return_position
        else:
            Logger.debug("Cannot backtrace trajectory-{:d} to frame-{:d}. Exceeded permitted backtrace interval.")
            return None

    def get_particle_by_time(self, time: int) -> Particle:
        """
        Get the particle object at the given time. Unlike get_position_by_time() which interpolates
        to the given time if it's outside the range of start_time and end_time, this function returns
        None if there's no particle at the given time.
        """
        if not self.__sorted:
            self.sort_particles()

        if time < self.start_time or time > self.end_time:
            Logger.debug(f"Specified time is outside the life time of trajectory-{self.id}: " + \
                         f"{time:4d} {self.start_time:4d} {self.end_time:4d}")
            return None

        if time in self.ptcl_time_map:
            return self.ptcl_time_map[time]

        Logger.debug(f"No particle exists for trajectory-{self.id} at the time: {time:4d}")
        return None

    def get_snapshots(self, start: int, end: int) -> List[Particle]:
        """Retrun a deep copy of lists of particles between the time interval (inclusive)."""
        snapshot = []
        for p in self.ptcls:
            if start <= p.time_frame and p.time_frame <= end:
                snapshot.append(copy.deepcopy(p))
        return snapshot

    def has_particle(self, time: int) -> bool:
        """
            Whether a particle already exists in this trajectory at the given time.
        """
        for p in self.ptcls:
            if p.time_frame == time:
                return True
        return False

    def get_particles(self) -> List[Particle]:
        return self.ptcls

    def get_start_particle(self) -> Particle:
        if self.ptcls is None: # Actually this should never be None.
            return None

        if not self.__sorted:
            self.sort_particles()
        return self.ptcls[0]

    def get_end_particle(self) -> Particle:
        if self.ptcls is None: # Actually this should never be None.
            return None

        if not self.__sorted:
            self.sort_particles()
        return self.ptcls[-1]

    def _compute_velocity_vectors(self) -> bool:
        """
        If the list of instant velocity vectors in not computed, initialize and compuate them.
        Short-circuit if they have already been computed.

        Note: It relies on the "__sorted" attribute to determine whether the non-empty vector
        is up-to-date or not.
        """
        if not self.__sorted:
            self.sort_particles() # Internally already called self.reset()

        if len(self.ptcls) <= 1:
            Logger.debug(f"Calculating velocities for trajectory-{self.id} with "
                           f"{len(self.ptcls)} particles.")
            return False

        if self.velocity_vectors is not None and not math.isnan(self.avg_velocity):
            return True # Already computed. Short circuit.

        self.velocity_vectors = np.zeros(shape=(len(self.ptcls) - 1, 2), dtype=np.float64)
        for i in range(0, len(self.ptcls) - 1):
            p1 = self.ptcls[i]
            p2 = self.ptcls[i + 1]
            if p1.time_frame == p2.time_frame:
                # Multiple particles exists for the same time frame.
                Logger.error("Fail to calculate velocity. Multiple particles " \
                            f"exist in this trajectory-{self.id} at the same time frame {p1.time_frame}")
                Logger.error(p1.short_rep())
                Logger.error(p2.short_rep())
                continue
            x1, y1 = p1.get_position()
            x2, y2 = p2.get_position()
            dt = p2.time_frame - p1.time_frame
            self.velocity_vectors[i][0] = (x2 - x1) / dt
            self.velocity_vectors[i][1] = (y2 - y1) / dt
        self.avg_velocity = np.average(np.linalg.norm(self.velocity_vectors, axis=1))
        return True

    def get_average_velocity(self) -> float:
        """
            Calculate and return the average velocity averaged over velocities at all time
            frames of the underlying particle.
        """
        if not self._compute_velocity_vectors():
            return float("nan")
        return self.avg_velocity

    def get_instant_velocity_max(self) -> float:
        """
        Calculate the maximum instantaneous velocity between two consecutive positions.
        """
        if not self._compute_velocity_vectors():
            return float("nan")
        return np.max(np.linalg.norm(self.velocity_vectors, axis=1))

    def get_velocity_angle_max(self) -> float:
        """
        Get the maximum angle between any two consecutive velocity vectors.
        """
        if not self._compute_velocity_vectors():
            return float("nan")
        elif len(self.velocity_vectors) < 2:
            Logger.debug("Cannot compute angle for one velocity vector.")
            return 0.0

        angles = np.zeros(len(self.velocity_vectors) - 1)
        for i in range(0, len(self.velocity_vectors) - 1):
            angles[i] = vector_angle(self.velocity_vectors[i],
                                     self.velocity_vectors[i + 1])
        if np.sum(np.isnan(angles)) == len(angles):
            Logger.debug(f"All velocity vectors are zero. Cannot get a max angle. {self.id}")
            return float("nan")
        return np.nanmax(angles) # exclude "nan"

    def get_velocity_angle_full(self) -> float:
        """
        Get the angle between the first non-zero velocity vectors from the start
        and the last non-zero velocity vector at the end.
        """
        if not self._compute_velocity_vectors():
            return float("nan")
        elif len(self.velocity_vectors) < 2:
            Logger.debug("Calculate angle for one velocity vector.")
            return 0.0

        norms = np.linalg.norm(self.velocity_vectors, axis=1)
        index_i = len(self.velocity_vectors) # first non-zero vector from start
        for i in range(0, len(norms)):
            if norms[i] > 0:
                index_i = i
                break
        index_j = -1
        for j in range(len(norms) - 1, -1, -1):
            if norms[j] > 0:
                index_j = j
                break

        if index_i >= index_j: # don't allow the same vector.
            Logger.debug(f"All velocity vectors are zero. Cannot get a full angle. {self.id}")
            return float("nan")

        return vector_angle(self.velocity_vectors[index_i],
                            self.velocity_vectors[index_j])


    def get_average_particle_size(self) -> float:
        """
        Average the particle size over all time frames.
        """
        sum = 0
        for p in self.ptcls:
            sum += p.get_size()
        return sum / len(self.ptcls)

    def get_shape(self) -> str:
        shapes = []
        for p in self.ptcls:
            shapes.append(p.get_shape())

        counts = Counter(shapes)
        traj_shape = counts.most_common(1)[0][0]

        if traj_shape == "N/A":
            return "undetermined"
        return traj_shape

    def reset(self):
        """
            Reset post-processed properties of the trajectory when new particles are
            appended to the trajectory. A reset might be needed becuase appending of
            this new particle could change these post-processing properties.
        """
        self.start_time = -1
        self.end_time = -1
        self.avg_velocity = float("nan")
        self.velocity_vectors = None
        self.__sorted = False

    def __str__(self) -> str:
        string = f"Trajectory: Particle id: {self.id:4d}; \t" + \
                 f"Start time: {self.get_start_time():5d}; \t" + \
                 f"End time: {self.get_end_time():5d}; \t" + \
                 f"Average size: {self.get_average_particle_size():9.4f}; \t" + \
                 f"Average velocity: {self.get_average_velocity():9.4f}; \t" + \
                 f"Max instantaneous velocity: {self.get_instant_velocity_max():9.4f}; \t" + \
                 f"Max angle between velocity vectors: {self.get_velocity_angle_max():9.4f}; \t" + \
                 f"Angle between start and end velocity vectors: {self.get_velocity_angle_full():9.4f}; \t" + \
                 f"Shape: {self.get_shape():12s}" \
                 #+ " " + str(self.velocity_vectors)
        return string