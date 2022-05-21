# -*- coding: utf-8 -*-
"""
Kalman class using opencv implementation
"""

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from xmot.mot.utils import cen2cor, cor2cen, costCalc, unionBlob, iom

class Blob:
    """
    Abstraction of identified particles in video, (i.e. unqiue particle).

    Attributes:
        idx: integer Particle ID.
        bbox: [x1, y1, x2, y2] Coordinates of upper left and lower right corners.
        color: [x, y, z] RGB color code of the particle.
        dead: <TODO>
        frames: [<integer>] List of frame ids the particle lives in.
        kalm: CV KalmanFilter The kalmanfilter tracking this particle.
    """
    def __init__(self, idx, bbox, mask):
        self.idx    = idx
        self.bbox   = bbox
        self.masks  = [mask]
        self.color  = np.random.randint(0,255,size=(3,))
        self.dead   = 0
        self.frames = []
        
        # Kalman object
        self.kalm  = cv.KalmanFilter(8, 4, 0)
        
        # transition matrix
        F = np.array([[1, 0, 0, 0, 1, 0, 0, 0], # cenx
                      [0, 1, 0, 0, 0, 1, 0, 0], # ceny
                      [0, 0, 1, 0, 0, 0, 1, 0], # w
                      [0, 0, 0, 1, 0, 0, 0, 1], # h
                      [0, 0, 0, 0, 1, 0, 0, 0], # vx
                      [0, 0, 0, 0, 0, 1, 0, 0], # vy
                      [0, 0, 0, 0, 0, 0, 1, 0], # w_dot
                      [0, 0, 0, 0, 0, 0, 0, 1]  # h_dot
                      ], dtype=np.float32)
        
        self.kalm.transitionMatrix = F
        
        # measurement matrix
        self.kalm.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        
        # process noise covariance
        self.kalm.processNoiseCov = 4.*np.eye(8, dtype=np.float32)
        
        # measurement noise covariance
        self.kalm.measurementNoiseCov = 4.*np.eye(4, dtype=np.float32)
        
        # Set posterior state
        state = list(cor2cen(self.bbox)) + [0, 0, 0,0 ]
        self.kalm.statePost = np.array(state, dtype=np.float32)
        
    def predict(self):
        state = self.kalm.predict()
        self.bbox = np.array(cen2cor(state[0], state[1], state[2], state[3]))
        return state
    
    def correct(self,measurement,mask):
        self.masks.append(mask)
        self.kalm.correct(measurement)
        
        # correct bbox
        state     = self.kalm.statePost
        self.bbox = np.array(cen2cor(state[0],state[1],state[2],state[3]))
        
    def statePost(self):
        return self.kalm.statePost

class MOT:
    def __init__(self, bbox, mask, fixed_cost=100., merge=False, merge_it=2, merge_th=50):
        self.total_blobs = 0
        self.cnt         = 0              # Frame id
        self.blobs       = []             # List of all blobs (idenfied particle)
        self.blobs_all   = []
        self.blolen      = len(bbox)      # Total number of blobs
        self.fixed_cost  = fixed_cost
        self.merge       = merge          # Flag: whether to merge bboxes
        self.merge_it    = merge_it
        self.merge_th    = merge_th
        
        # assign a blob for each box
        for i in range(self.blolen):
            # assign a blob for each bbox
            self.total_blobs += 1
            b = Blob(self.total_blobs,bbox[i], mask[i])
            b.frames.append(self.cnt)
            self.blobs.append(b)
        
        # optional box merge
        if merge:
            self.__merge()
    
    def step(self, bbox, mask):
        """
        Add bboxes of a frame and create/merge/delete blobs.
        """
        # advance cnt
        self.cnt += 1
        
        # make a prediction for each blob
        self.__pred()
        
        # calculate cost and optimize using the Hungarian algo
        blob_ind = self.__hungarian(bbox)
        
        # Update assigned blobs if exists else create new blobs
        new_blobs = self.__update(bbox,blob_ind, mask)
        
        # Blobs to be deleted
        ind_del = self.__delBlobInd(bbox, blob_ind)
        
        # Delete blobs
        self.__delBlobs(ind_del)
        
        # Add new blobs
        self.blobs += new_blobs
        
        # Optional merge
        if self.merge:
            self.__merge()
        
        self.blolen = len(self.blobs)
    
    def __pred(self):
        # predict next position
        for i in range(self.blolen):
            self.blobs[i].predict()
    
    def __hungarian(self, bbox):
        cost = costCalc(bbox, self.blobs, self.fixed_cost)
        box_ind, blob_ind = linear_sum_assignment(cost)
        return blob_ind
    
    def __update(self,bbox,blob_ind,mask):
        boxlen = len(bbox)
        new_blobs = []
        for i in range(boxlen):
            m   = np.array(cor2cen(bbox[i]), dtype=np.float32)
            ind = blob_ind[i]
            if ind < self.blolen:
                self.blobs[ind].correct(m, mask[i])
            else:
                self.total_blobs += 1
                b = Blob(self.total_blobs,bbox[i],mask[i])
                b.frames.append(self.cnt)
                new_blobs.append(b)
        return new_blobs
    
    def __delBlobInd(self, bbox, blob_ind):
        # get unassigned blobs
        boxlen  = len(bbox)
        ind_del = []
        for i in range(boxlen,len(blob_ind)):
            if blob_ind[i] < boxlen:
                ind_del.append(blob_ind[i])
        
        return ind_del
    
    def __delBlobs(self,ind_del):
        # sort to start removing from the end
        ind_del.sort(reverse=True)
        for ind in ind_del:
            if self.blobs[ind].dead > 2:
                self.blobs_all.append(self.blobs[ind])
                self.blobs.pop(ind)
            else:
                self.blobs[ind].dead += 1
    
    def __merge(self):
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