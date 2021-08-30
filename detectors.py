# -*- coding: utf-8 -*-
"""
various bounding box detection schemes
"""

import numpy as np
import cv2 as cv

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
    def __init__(self,fname, history=100, varThreshold=40, it_closing=1, minArea=20):
        self.cap        = cv.VideoCapture(fname)
        self.it_closing = it_closing
        self.minArea    = minArea
        self.gmm        = cv.createBackgroundSubtractorMOG2(history=history,
                                                            varThreshold=varThreshold)
        print("Training GMM")
        self.__train()
        
    def getBbox(self,img):
        gray    = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
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
            self.gmm.apply(gray)
        self.cap.release()
    
        

# # train gmm
# gmm   = 
# while(True):
#     _, img = cap.read()
#     if img is None: break
#     gray   = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     gmm.apply(gray)
# cap.release()