# -*- coding: utf-8 -*-
"""
various bounding box detection schemes
"""
import os

import numpy as np
import cv2 as cv
from src.logger import Logger

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class Canny:
    def __init__(self, th1=50, th2=100, minArea=20, it_closing=1):
        self.th1        = th1
        self.th2        = th2
        self.minArea    = minArea
        self.it_closing = it_closing

    def getBbox(self, img):
        # convert to grayscale
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        # Canny edge detection
        edge = cv.Canny(gray, self.th1, self.th2)

        # Morphological transformation: closing
        kernel  = np.ones((8,8), dtype=np.uint8)
        closing = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel, iterations=self.it_closing)

        # Contours
        contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Bounding rectangle
        bbox = []
        for temp in contours:
            area = cv.contourArea(temp)
            if area>self.minArea:
                x,y,w,h = cv.boundingRect(temp)
                bbox.append([x,y,x+w,y+h])

        return bbox

class GMM:
    def __init__(self, fname, crop, history=100, varThreshold=40, it_closing=1, minArea=20):
        """

        Attributes:
            fanme        : String     Path to the video
            crop         : (int, int) Sizes in x and y dimension for cropping each video frame. 
            history      : int        Length of history 
            varThreshold : int        Threshold of pixel background identification in MOB2 of cv.
            it_closing   : int        Parameter of cv.morphologyEx
            minArea      : int        Minimal area of bbox to be considered as a valid particle.
        """
        self.cap        = cv.VideoCapture(fname)
        self.it_closing = it_closing
        self.minArea    = minArea
        self.crop       = crop
        self.gmm        = cv.createBackgroundSubtractorMOG2(history=history,
                                                            varThreshold=varThreshold)
        Logger.detail("Training GMM ...")
        self.__train()

    def getBbox(self,img):
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        mask = self.gmm.apply(gray)
        # Morphological transformation: closing
        kernel  = np.ones((8,8), dtype=np.uint8)
        closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=self.it_closing)

        # Contours
        contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Bounding rectangle
        bbox = []
        for temp in contours:
            area = cv.contourArea(temp)
            if area>self.minArea:
                x,y,w,h = cv.boundingRect(temp)
                bbox.append([x,y,x+w,y+h])
        return bbox

    def __train(self):
        while(True):
            _, img = self.cap.read()
            if img is None: break
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            gray = gray[0:self.crop[0],0:self.crop[1]]
            self.gmm.apply(gray)
        self.cap.release()