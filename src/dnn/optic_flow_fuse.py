#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuse DNN predictions with optical flow
"""

import numpy as np
import scipy.io as sio
import os
import cv2 as cv
import argparse

from utils import *

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--th_speed", type=float, default=0.2,
                        help="Speed threshold for merging bounding boxes and segmentations")
    parser.add_argument("-d", "--th_dist", type=float, default=2,
                        help="Center distance threshold for merging bounding boxes and segmentations")
    parser.add_argument("-m", "--mname", type=str, default="test",
                        help="name of the video file to be processed")
    parser.add_argument("-c", "--crop", type=int, default=512,
                        help="Crop each frame. 0 (default) for no cropping")
    parser.add_argument("-r", "--rad", type=float, default=1,
                        help="Radius to measure the velocity from bbox center")
    args = parser.parse_args()

    # read video file
    cwd = os.getcwd()  # get current working directory
    cap = cv.VideoCapture(cwd + "/" + args.mname + ".mp4")
    th_speed = args.th_speed  # relative speed threshold for mask and bbox union
    th_dist = args.th_dist  # max distance between centers for merging boxes
    crop = args.crop
    r = args.rad

    fname = "/video/boxes/frame"
    sname = "/video/fused_boxes/"
    try:
        os.mkdir(cwd + sname)
    except:
        print("folder already exists, overwriting contents ...")

    # Get first frame (needed for optical flow)
    idx = 0
    ret, frame1 = cap.read()
    prv = cv.cvtColor(frame1[0:crop, 0:crop, :], cv.COLOR_BGR2GRAY)

    # crop 0 is no crop
    if crop == 0: crop = prv.shape[0]

    # Meshgrid for indexing
    yy, xx = np.ogrid[0:crop, 0:crop]

    while (1):
        # Calculate flow
        ret, frame2 = cap.read()
        if frame2 is None: break

        nxt = cv.cvtColor(frame2[0:crop, 0:crop, :], cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prv, nxt, None, pyr_scale=0.5, levels=5,
                                           winsize=15, iterations=3, poly_n=5,
                                           poly_sigma=1.2, flags=0)
        prv = nxt
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Read DNN predictions
        idx += 1
        dat = sio.loadmat(cwd + fname + str(idx) + ".mat")
        bbox = dat["bbox"]
        mask = dat["mask"].astype(np.bool)
        npar = len(bbox)

        # Average speed for each particle
        cen = np.zeros((npar, 2))  # center position
        speed = np.zeros((npar,))  # average speed around the center
        for j in range(npar):
            # cen[j,:] = np.array([bbox[j,0] + bbox[j,2],          # x
            #                      bbox[j,1] + bbox[j,3]]) / 2.    # y
            # indxs = (xx-cen[j,0])**2 + (yy-cen[j,1])**2 <= r*r
            # indxs = np.logical_and(indxs, mask[j,...])
            speed[j] = np.mean(mag[mask[j, :, :]])

        # normalize speeds (useful for plotting later)
        max_speed = np.max(speed)
        speed = speed / max_speed

        # Fuse if speed is similar
        mask_new, bbox_new, speed_new = mergeBoxes(mask, bbox, speed, mag,
                                                   max_speed, th_speed, th_dist, 2)

        # Save new bounding boxes
        sio.savemat(cwd + sname + "frame" + str(idx) + ".mat",
                    {"mask": mask_new, "bbox": bbox_new, "speed": speed_new})
        print(idx)

        # Draw boxes
        img1 = drawBoxSpeed(frame2[0:crop, 0:crop, :].copy(), bbox, speed)
        cv.imshow("Before", img1)

        img2 = drawBoxSpeed(frame2[0:crop, 0:crop, :].copy(), bbox_new, speed_new)
        cv.imshow("After", img2)

        # k = cv.waitKey(30) & 0xff
        # if k == 27:
        #     break
        k = cv.waitKey(0) & 0xff
        if k == 27:
            break

    # release resources
    cap.release()
    cv.destroyAllWindows()