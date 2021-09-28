# -*- coding: utf-8 -*-
"""
Utility functions
"""

import numpy as np
import cv2 as cv
from logger import Logger

def imosaic(img_list, size=None, gray=False):
    '''
    Create a mosaic image from a nested list of images

    Parameters
    ----------
    img_list : Nested list of images. Ex: [[img11,img12],[img21, img22]]
    gray: draw grayscale

    Returns
    -------
    Final image mosaic (H, W, C)

    '''
    
    # concat columns in every row
    rows = []
    for row in img_list:
        proc_list = []
        for temp in row:
            nc = len(temp.shape) # 2 for gray image, 3 for color
            
            # if color image but need to convert to gray
            if nc == 3 and gray:
                proc_list.append(cv.cvtColor(temp, cv.COLOR_BGR2GRAY))
            # if gray image but need to convert to color
            elif nc == 2 and not gray:
                proc_list.append(cv.cvtColor(temp, cv.COLOR_GRAY2BGR))
            # otherwise just append
            else:
                proc_list.append(temp)
        rows.append(np.concatenate(proc_list, axis=1))
    
    # concat all the rows
    output = np.concatenate(rows, axis=0)
    
    # resize to desired output size
    if size is not None:
        output = cv.resize(output, size)
    return output


def drawBox(img, bbox, color=(0,0,255)):
    '''
    Draw rectangular bounding box on a given image

    Parameters
    ----------
    img : Image
    bbox : List of rectangular bounding boxes to be drawn format (x1, y1, x2, y2)

    Returns
    -------
    img : Modified image

    '''
    for j in range(len(bbox)):
        x1,y1,x2,y2 = bbox[j]
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    return img

def drawBlobs(img, blobs):
    for j in range(len(blobs)):
        x1,y1,x2,y2 = blobs[j].bbox
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        color       = blobs[j].color
        color       = (int(color[0]), int(color[1]), int(color[2]))
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

def writeBlobs(blobs, file, frameID):
    with open(file, "a") as f:
        for i in range(len(blobs)):
            #Logger.debug("Number of frames for this blob: {:d}".format(
            #    len(blobs[i].frames)))
            x1,y1,x2,y2 = blobs[i].bbox
            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
            w = x2 - x1
            h = y2 - y1
            idx = blobs[i].idx
            frames = blobs[i].frames
            f.write(("{:4d}, " * 7).format(x1, y1, x2, y2, w, h, idx))
            f.write("{:4d}".format(frameID))
            #f.write(",".join([str(frame) for frame in frames]))
            f.write("\n")

def findClosestBox(x,y,bbox):
    '''
    Given coordinates x,y and a list of bounding boxes,
    find the box that is closest to x,y

    Parameters
    ----------
    x : (width) x coordinate of box search
    y : (height) y coordinate of box search
    bbox : List of bounding boxes each with (x1,y1,x2,y2) format

    Returns
    -------
    box : box that best matches (x,y)
    out : index of the output box

    '''
    dist = 10000
    box  = [0,0,0,0]
    idx  = 0
    out  = -1
    for b in bbox:
        cenx = (b[0] + b[2])/2.
        ceny = (b[1] + b[3])/2.
        dist_new = np.sqrt((x-cenx)**2 + (y-ceny)**2)
        if dist_new < dist:
            dist = dist_new
            box  = b
            out  = idx
        idx += 1
    return box, out

def cor2cen(bbox):
    '''
    Convert bounding box edge coordinates of the form (x1,y1,x2,y2) to
    center coordinates of the form (cenx, ceny, w, h)
    '''
    cenx = (bbox[0] + bbox[2])/2.
    ceny = (bbox[1] + bbox[3])/2.
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    return cenx,ceny,w,h

def cen2cor(cenx,ceny,w,h):
    '''
    Convert bounding box center coordinates of the form (cenx, ceny, w, h) to
    edge coordinates of the form (x1,y1,x2,y2)
    
    '''
    hw = w/2.
    hh = h/2.
    
    x1 = cenx - hw
    x2 = cenx + hw
    y1 = ceny - hh
    y2 = ceny + hh
    
    return x1,y1,x2,y2

def costCalc(bbox, blobs, fixed_cost=100.):
    boxlen = len(bbox)
    blolen = len(blobs)
    
    # size of cost array twice the largest
    # that way every blob can be deleted and new bbox can be created
    length = 2*max(boxlen, blolen)
    cost = np.ones((length,length), dtype=np.float64) * fixed_cost
    
    # Calculate cost
    for i in range(boxlen):
        for j in range(blolen):
            cenx1,ceny1,w1,h1 = cor2cen(bbox[i])
            cenx2,ceny2,w2,h2 = cor2cen(blobs[j].bbox)
            
            # eucledian distance
            cost[i][j] = np.sqrt( (cenx1 - cenx2)**2 + (ceny1 - ceny2)**2 )
    
    return cost

def iou(bbox1, bbox2):
    '''
    Intersection over union for two bounding boxes
    
    see: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    Parameters
    ----------
    bbox1 : Bounding box (x1,y1,x2,y2)
    bbox2 : Bounding box (x1,y1,x2,y2)

    Returns
    -------
    Intersection over union [0, 1]

    '''
    x1 = max(bbox1[0],bbox2[0])
    y1 = max(bbox1[1],bbox2[1])
    x2 = min(bbox1[2],bbox2[2])
    y2 = min(bbox1[3],bbox2[3])
    
    aint = max(0, x2 - x1 + 1.) * max(0, y2 - y1 + 1.)
    a1   = (bbox1[2] - bbox1[0] + 1.) * (bbox1[3] - bbox1[1] + 1.)
    a2   = (bbox2[2] - bbox2[0] + 1.) * (bbox2[3] - bbox2[1] + 1.)

    iou = aint/(a1 + a2 - aint)
    
    return iou

def iom(bbox1, bbox2):
    '''
    Intersection over min for two bounding boxes
    
    see: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    Parameters
    ----------
    bbox1 : Bounding box (x1,y1,x2,y2)
    bbox2 : Bounding box (x1,y1,x2,y2)

    Returns
    -------
    Intersection over union [0, 1]

    '''
    x1 = max(bbox1[0],bbox2[0])
    y1 = max(bbox1[1],bbox2[1])
    x2 = min(bbox1[2],bbox2[2])
    y2 = min(bbox1[3],bbox2[3])
    
    aint = max(0, x2 - x1 + 1.) * max(0, y2 - y1 + 1.)
    a1   = (bbox1[2] - bbox1[0] + 1.) * (bbox1[3] - bbox1[1] + 1.)
    a2   = (bbox2[2] - bbox2[0] + 1.) * (bbox2[3] - bbox2[1] + 1.)

    iom = aint/min(a1, a2)
    
    return iom

def unionBlob(blob1, blob2):
    '''
    Merge two blobs

    Parameters
    ----------
    blob1 : Main blob
    blob2 : Blob that merges with the main

    Returns
    -------
    Main blob
    '''
    # Average posteior state
    blob1.kalm.statePost = (blob1.kalm.statePost + blob2.statePost())/2.
    
    # update bbox
    state      = blob1.kalm.statePost
    blob1.bbox = np.array(cen2cor(state[0],state[1],state[2],state[3]))
    
    # stub
    # modify frames and dead 
    return blob1