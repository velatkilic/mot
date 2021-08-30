# -*- coding: utf-8 -*-
"""
Human in the loop
"""

import cv2 as cv
import numpy as np
import os
import argparse

from utils import imosaic, drawBox, findClosestBox, drawBlobs
from kalman import MOT
from detectors import Canny, GMM

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("-n","--name", type=str, default="playback_test300AlZr.avi", 
                    help="name of the video file to be processed")

parser.add_argument("-c","--control", type=bool, default=True,
                    help="If true, human in the loop mode is enabled")

parser.add_argument("-d","--model", type=str, default="gmm",
                    help="Object detector model. Currently available options\
                        are: gmm and canny" )
args = parser.parse_args()

vidname   = args.name                   # video file name
hin       = args.control                # Human in the loop?
cwd       = os.getcwd()                 # current working directory
fname     = os.path.join(cwd,vidname)   # join cwd with videoname
cap       = cv.VideoCapture(fname)      # video capture object for reading frames

if args.model == "canny":
    det = Canny()
elif args.model == "gmm":
    det = GMM(fname)
else:
    print("Invalid model name. Pick either gmm or canny")

# Mouse events for human in the loop control
drawing = False
def mouse_event(event,x,y,flags,param):
    global cont, img, bbox, drawing, rx1, ry1, rx2, ry2
    
    # left button selects image
    if event==cv.EVENT_LBUTTONDOWN:
        # find closest bounding box
        [xb,yb,x2b,y2b], idx = findClosestBox(x,y,bbox)
        
        # highlight found bounding box
        cont = cv.rectangle(cont, (xb, yb), (x2b, y2b), (0,255,0), 2)
        cv.imshow("Frame", cont)
        
        # wait for a key
        k = cv.waitKey(0)
        # if key is 'd' then remove the bounding box
        if k==ord('d'):
            bbox.pop(idx)
        cont = drawBox(img.copy(), bbox)
        cv.imshow("Frame", cont)
    elif event==cv.EVENT_RBUTTONDOWN:
        if not drawing:
            rx1,ry1   = x,y
            drawing = True
        else:
            rx2,ry2 = x,y
            drawing = False
            bbox.append([rx1,ry1,rx2,ry2])
            cont = drawBox(img.copy(), bbox)
            cv.imshow("Frame", cont)
    elif event==cv.EVENT_MOUSEMOVE:
        if drawing:
            cont_new = cv.rectangle(cont.copy(), (rx1, ry1), (x,y), (0,0,255), 1)
            cv.imshow("Frame", cont_new)

# Object detection and kalman
cnt = 0
while(True):
    # read a single frame
    _, img = cap.read()
    
    if img is None: break
    
    # edge = gmm.apply(gray)
        # _, edge = cv.threshold(edge,50,255,cv.THRESH_BINARY)
    
    bbox = det.getBbox(img)
    
    # Draw bounding boxes
    cont = drawBox(img.copy(), bbox)
    
    # Show final image
    cv.imshow("Frame", cont)
    cv.setMouseCallback('Frame',mouse_event)
    
    # # Show debug info
    # W,H,_ = img.shape
    # zeros = np.zeros((W,H,3))
    # debug = imosaic([[img,edge],[closing, cont]],size=(1000,1000))
    
    # cv.imshow("Debug Info", debug)
    
    # Wait for a key
    k = cv.waitKey(0)
    if k == 27:
        break
    
    # Kalman tracking
    if cnt == 0:
        mot = MOT(bbox)
    else:
        mot.step(bbox)
        
    cnt  += 1
    
    img_kalman = drawBlobs(img.copy(), mot.blobs)
    cv.imshow("Kalman", img_kalman)
    
    k = cv.waitKey(0)
    if k == 27:
        break

# detroy cv objects
cap.release()
cv.destroyAllWindows()