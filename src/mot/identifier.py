# -*- coding: utf-8 -*-
"""
Human in the loop
"""

import cv2 as cv
import numpy as np

from src.mot.utils import imosaic, drawBox, findClosestBox, drawBlobs, writeBlobs
from src.mot.kalman import MOT
from src.mot.detectors import Canny, GMM
from src.logger import Logger


def identify(fname, model, imgOutDir, blobsOutFile, control=False, crop=(512, 512)):
    """
    Identify particles using specified model.

    Attributes:
        fname         : String  Path to the video.
        model         : String  Model used to identify particle.
        imgOutDir     : String  Output folder of images with bounding boxes.
        blobsOutFile  : String  Output file for info of each identidied particle.
        control       : Boolean Whether have human-in-the-loop control.
        crop          : (int, int) Cropping sizes in x and y dimension.
    """
    hin   = control                     # Flag: Human in the loop
    cap   = cv.VideoCapture(fname)      # video capture object for reading frames

    if model == "canny":
        det = Canny()
    elif model == "gmm":
        det = GMM(fname, crop)
    else:
        Logger.warning("Invalid model name. Pick either gmm or canny")
        det = GMM(fname,crop)

    # Mouse events for human in the loop control
    drawing = False

    # Object detection and kalman
    Logger.detail("Detecting particles ...")
    cnt = 0
    while(True):
        # read a single frame
        _, img = cap.read()
        
        if img is None: break
        
        # if crop needed
        img = img[0 : crop[0], 0 : crop[1], :]
        
        bbox = det.getBbox(img)
        
        # Draw bounding boxes
        cont = drawBox(img.copy(), bbox)
        
        # Show final image
        #cv.imshow("Frame", cont)
        cv.imwrite("{:s}/{:s}_{:d}.jpg".format(imgOutDir, model, cnt), cont)
        if hin: cv.setMouseCallback('Frame', mouse_event)
        
        # Wait for a key
        k = cv.waitKey(30)
        if k == 27: # ESC to stop
            break
        
        # Kalman tracking
        if cnt == 0:
            mot = MOT(bbox)
        else:
            mot.step(bbox)
            
        cnt  += 1
        
        img_kalman = drawBlobs(img.copy(), mot.blobs)
        writeBlobs(mot.blobs, blobsOutFile, mot.cnt)
        #cv.imshow("Kalman", img_kalman)
        #k = cv.waitKey(30)
        #if k == 27: # ESC to stop
        #    break
    # detroy cv objects
    cap.release()
    cv.destroyAllWindows()

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