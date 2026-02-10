# -*- coding: utf-8 -*-
"""
Kalman class using opencv implementation
"""

import cv2 as cv
import numpy as np
from typing import List, Any
from scipy.optimize import linear_sum_assignment
from xmot.mot.utils import cen2cor, cor2cen, costMatrix, unionBlob, iom, mask_to_cnt, cnt_to_mask
from xmot.logger import Logger

class Blob:
    """
    Abstraction of identified particles in video, (i.e. unqiue particle).

    Attributes:
        idx:    integer:    Particle ID, starting from 1.
        state:  List[int]:  [centroid_x, centroid_y, width, height]: centroid coordinates and dimension of bbox
                            of the particle contour
        color:  List[int]:  [x, y, z] RGB color code of the particle.
        dead:   int:        Number of consecutive frames the particle has not been detected.
        frames: List[int]:  List of frame IDs that the particle are considered existing.
        kalm:   KalmanFilter: The kalmanfilter object tracking this particle.
        masks:  Dict[int, np.ndarray]: List of masks with dimension (1, img_height, img_width) of the
                                       particle at each frame. For frames this particle is not detected
                                       yet alive, there is an empty array as a place holder.
        contours: Dict[int, np.ndarray]: [num_frames, N, 1, 2]. For each frame, the list has a ndarray
                                         of shape (N, 1, 2). For frames this particle is not detected
                                         yet alive, there is an empty array as a place holder.

        (Deprecated) bbox: [x1, y1, x2, y2] Coordinates of upper left and lower right corners.
    """
    def __init__(self, idx: int, frame_id: int, state: np.ndarray, mask: np.ndarray):
        self.idx    = idx
        #self.bbox   = bbox
        self.state = state
        self.masks  = {frame_id: mask}
        self.color  = np.random.randint(0,255,size=(3,))
        self.dead   = 0
        self.frames = [frame_id]
        self.contours = {frame_id: mask_to_cnt(mask)[0]} # The first element of the return tuple is the contour.

        # Kalman object
        self.kalm  = cv.KalmanFilter(8, 4, 0)

        # transition matrix
        F = np.array([[1, 0, 0, 0, 1, 0, 0, 0], # centroid x of the contour
                      [0, 1, 0, 0, 0, 1, 0, 0], # centroid y of the contour
                      [0, 0, 1, 0, 0, 0, 1, 0], # w
                      [0, 0, 0, 1, 0, 0, 0, 1], # h
                      [0, 0, 0, 0, 1, 0, 0, 0], # v_centroid_x
                      [0, 0, 0, 0, 0, 1, 0, 0], # v_centroid_y
                      [0, 0, 0, 0, 0, 0, 1, 0], # w_dot
                      [0, 0, 0, 0, 0, 0, 0, 1]  # h_dot
                      ], dtype=np.float32)

        self.kalm.transitionMatrix = F

        # measurement matrix
        # Can only measure center_x, center_y, w, h. Therefore, first dimension is 4.
        self.kalm.measurementMatrix = np.eye(4, 8, dtype=np.float32)

        # process noise covariance
        self.kalm.processNoiseCov = 4.*np.eye(8, dtype=np.float32)

        # measurement noise covariance
        self.kalm.measurementNoiseCov = 4.*np.eye(4, dtype=np.float32)

        # Set posterior state
        #state = list(cor2cen(self.bbox)) + [0, 0, 0, 0]
        _state = self.state + [0, 0, 0, 0]
        self.kalm.statePost = np.array(_state, dtype=np.float32)

    def predict(self):
        state = self.kalm.predict()
        #self.bbox = np.array(cen2cor(state[0], state[1], state[2], state[3]))
        state = np.squeeze(state) # predict() returns an array of shape (8, 1)
        self.state = np.array([state[0], state[1], state[2], state[3]])
        return state

    def correct(self, frame_id: int, measurement: np.ndarray[np.float32], mask: np.ndarray[Any, np.dtype[np.bool_]]):
        """Correct the state with new measurement.

        Args:
            measurement: [center_x, center_y, w, h] Measured state. When measurement is None, add
                         an empty mask and contour.
            mask: binary mask of the particle. Cannot be empty array.
        """
        if measurement is None:
            # This blob is not detected in the latest frame. Temporarily keep it alive.
            # The mask must be non-empty for the first detected frame.
            img_height, img_width = self.masks[self.frames[0]].shape[-2:]
            self.masks[frame_id] = np.array([], dtype=bool).reshape([0, img_height, img_width])
            self.contours[frame_id] = np.array([], dtype=np.int32).reshape([0, 1, 2]) # '0' for 0 particles
            return

        # Correct bbox with the state updated by Kalman from both estimation and measurement.
        self.kalm.correct(measurement)
        state     = self.kalm.statePost
        #self.bbox = np.array(cen2cor(state[0], state[1], state[2], state[3]))
        self.state = np.array([state[0], state[1], state[2], state[3]])

        # Note! The bbox might not enclose the mask since the bbox has been modified by the Kalman filter.
        self.masks[frame_id] = mask
        cnt, _img_height, _img_width, = mask_to_cnt(mask)
        self.contours[frame_id] = cnt

    def statePost(self):
        return self.kalm.statePost

    def get_bbox(self, frame_id = -1):
        """
        Get bbox from masks for the given frame. If frame_id is not given, return the bbox
        for the last detected frame. If frame_id is not in the list of frames for which the
        particle is detected, print a warning and return None.

        Return:
            Coordinates of the top-left and bottom-right corner.
        """
        if frame_id >= 0 and frame_id not in self.frames:
            Logger.warning(f'Particle {self.idx} does not exist in frame {frame_id}!')
            return None

        frame_id = frame_id if frame_id > 0 else self.frames[frame_id] # e.g. -1

        cnt, _height, _width = mask_to_cnt(self.masks[frame_id])
        x, y, w, h = cv.boundingRect(cnt)

        return x, y, x+w, y+h

    def delete_frame(self, frame_id):
        """
        Delete the frame ID from the frame list and corresponding mask.
        """
        if frame_id in self.frames:
            self.frames.remove(frame_id)

        if frame_id in self.masks:
            del self.masks[frame_id]
            del self.contours[frame_id]

class MOT:

    # Number of frames allow for a blob to be undetected before dropping it from tracking.
    UNDETECTION_THRESHOLD = 2

    def __init__(self, states, mask, fixed_cost=100., merge=False, merge_it=2, merge_th=50):
        """ Constructor.

        variables tracked by the Kalman filter. For now, the state contains the coordinates
        since centroid is not the middle of the bounding box.
        """
        self.frame_id    = 0              # Current frame id
        self.blobs       = []             # List of currently tracked blobs (idenfied particle)
        self.blolen      = len(states)    # Total number of blobs currently tracked.
        self.blobs_all   = []             # List of all blobs, including deleted ones.
        self.total_blobs = 0
        self.fixed_cost  = fixed_cost     # Basic cost in the cost matrix for assignment problem.
        self.merge       = merge          # Flag: whether to merge bboxes
        self.merge_it    = merge_it       # Iteration to operate the merge
        self.merge_th    = merge_th

        # assign a blob for each box
        for i in range(self.blolen):
            # assign a blob for each bbox
            self.total_blobs += 1
            # b = Blob(self.total_blobs, bbox[i], mask[i])
            b = Blob(self.total_blobs, self.frame_id, states[i], mask[i])
            self.blobs.append(b)
            self.blobs_all.append(b)

        # optional box merge
        # if merge:
        #    self.__merge()

    def step(self, states, mask):
        """
        Add bboxes of a frame and create/merge/delete blobs.
        """
        # advance the current frame_id
        self.frame_id += 1

        # make a prediction for each blob
        # Even for the blobs that haven't bee detected in last frame, but kept alive temporarily,
        # their position is updated by its Kalman Filter.
        self.__pred()

        # calculate cost and optimize using the Hungarian algo
        # When bbox has a length of zero (no particle in this frame), the assignment
        # retains its original order.
        blob_ind = self.__hungarian(states)

        # Update assigned blobs if exist. Otherwise, create new blobs
        new_blobs = self.__update(states, blob_ind, mask)  # Could be empty

        # Blobs to be deleted
        ind_del = self.__delBlobInd(states, blob_ind)

        # Delete blobs
        self.__delBlobs(ind_del)

        # Add new blobs
        self.blobs += new_blobs
        self.blobs_all += new_blobs
        self.total_blobs += len(new_blobs)

        # Optional merge
        # if self.merge:
        #    self.__merge()

        self.blolen = len(self.blobs)

    def __pred(self):
        # predict next position
        for i in range(self.blolen):
            self.blobs[i].predict()
            self.blobs[i].frames.append(self.frame_id) # Even include the "dead" frames.

    def __hungarian(self, states):
        """
        Return the ids of the existing blobs that matches the new bboxes in the new frame.
        """

        cost = costMatrix(states, self.blobs, fixed_cost=self.fixed_cost)
        # Default is to minimize the cost.
        box_ind, blob_ind = linear_sum_assignment(cost)
        return blob_ind

    def __update(self, states, blob_ind, mask):
        boxlen = len(states)
        new_blobs = []
        for i in range(boxlen):
            #m   = np.array(cor2cen(bbox[i]), dtype=np.float32)
            measurement = np.array(states[i], dtype=np.float32)
            ind = blob_ind[i]
            if ind < self.blolen:  # Case 1: Detected bbox match one of the existing blob.
                self.blobs[ind].correct(self.frame_id, measurement, mask[i])
                self.blobs[ind].dead = 0  # Recount the number of undetected frames.
            else:  # Case 2: Detected bbox don't match any of the existing blob.
                # blob.idx starts from 1.
                b = Blob(self.total_blobs + len(new_blobs) + 1, self.frame_id, states[i], mask[i])
                new_blobs.append(b)

        # Case 3: For the tracked but not detected particles, add empty mask and contour to Blob.
        for i in range(boxlen, len(blob_ind)):
            if blob_ind[i] < self.blolen:
                ind = blob_ind[i]
                self.blobs[ind].correct(self.frame_id, None, np.array([]))

        return new_blobs

    def __delBlobInd(self, states, blob_ind):
        # get unassigned blobs
        boxlen  = len(states)
        ind_del = []
        # Note: here, we need to iterate all the blob_ind, rather than only self.blolen.
        # Considering the case that there are equal number of detected particles in the new
        # frame and tracked particles by the Kalman filters. One tracked particle is not
        # detected any more. Its position would be boxlen+1, or equivalently self.blolen+1.
        for i in range(boxlen, len(blob_ind)):
            if blob_ind[i] < self.blolen:
                # Existing blob with blob_ind[i] does not match with any of the bboxes
                # in the new frame. Otherwise, blob_ind[i] should be the id of that new bbox,
                # which is smaller than boxlen.
                ind_del.append(blob_ind[i])

        return ind_del

    def __delBlobs(self, ind_del):
        # sort first and then start removing from the end
        ind_del.sort(reverse=True)
        for ind in ind_del:
            self.blobs[ind].dead += 1
            if self.blobs[ind].dead > MOT.UNDETECTION_THRESHOLD:
                # Keep trajectory alive for UNDETECTION_THRESHOLD to deal with flickering phenomenon.
                #self.blobs_all.append(self.blobs[ind])
                _blob = self.blobs.pop(ind)
                # Remove the last frame id from the frame list, so only two undetected frames
                # can exist in the frame list.
                _blob.delete_frame(self.frame_id)

                # When ending a blob, remove the allowance frames added to keep it temporarily
                # alive for MOT.UNDETECTION_THRESHOLD frames. So the blob can end properly without
                # creating ghost particles.
                for i in range(1, MOT.UNDETECTION_THRESHOLD + 1):
                    _blob.delete_frame(self.frame_id - i)

    def __merge(self):
        """
        (Deprecated) A bbox merge strategy based on location and velocity information from Kalman Filters.
        TODO: Handle the change of particle IDs when blobs are merged.
        """
        for i in range(self.merge_it):
            cursor_left  = 0
            cursor_right = 0
            length       = len(self.blobs)
            while(cursor_left < length):
                cursor_right = cursor_left + 1
                while(cursor_right < length):
                    # Get posterior states
                    state1    = self.blobs[cursor_left].statePost()
                    state2    = self.blobs[cursor_right].statePost()

                    # parse state vectors
                    cenx1,ceny1,w1,h1,vx1,vy1,_,_ = state1
                    cenx2,ceny2,w2,h2,vx2,vy2,_,_ = state2

                    # Metrics
                    dist    = np.sqrt( (cenx1-cenx2)**2 + (ceny1-ceny2)**2 )
                    dMetric = (dist**2)/(h1*w1) + (dist**2)/(h2*w2)
                    vMetric = np.sqrt( (vx1-vx2)**2 + (vy1-vy2)**2 )
                    iMetric = iom(self.blobs[cursor_left].bbox, self.blobs[cursor_right].bbox)

                    # merge
                    if vx1 == 0 and vx2 == 0 and vy1 == 0 and vy2 == 0:
                        mcon = iMetric>0.1
                    else:
                        mcon = (dMetric<1. or iMetric>0.05) and vMetric<2.
                        # mcon = (iMetric>0.05) and vMetric<1.

                    if mcon:
                        # merge blobs
                        blob1 = self.blobs[cursor_left]
                        blob2 = self.blobs[cursor_right]
                        self.blobs[cursor_left]  = unionBlob(blob1, blob2)

                        # pop merged data from lists
                        self.blobs.pop(cursor_right)
                        length = length - 1 # adjust length of the list
                    else:
                        cursor_right = cursor_right + 1
                cursor_left = cursor_left + 1

        # update blob length
        self.blolen = len(self.blobs)