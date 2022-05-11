import cv2 as cv
from xmot.mot.utils import drawBox, drawBlobs, writeBlobs, mergeBoxes
from xmot.mot.kalman import MOT
from xmot.mot.detectors import DNN
from xmot.datagen.bead_gen import bead_data_to_file, BeadDatasetFile, collate_fn
from xmot.datagen.style_data_gen_mask import StyleDatasetGen
import os
from xmot.logger import Logger
from torch.utils.data import DataLoader
import numpy as np

def identify(dset, imgOutDir, blobsOutFile, modelType="DNN", model=None, train_set=None, device="cuda:0", num_workers=0):
    """
    Identify particles using specified model.

    Attributes:
        dset          : Dataset  Instance of the video wrapper class Dataset.
        imgOutDir     : String  Output folder of video frames with detected bounding boxes.
        blobsOutFile  : String  Output file for info of each identified particle.
        modelType     : String  Type of detection model: DNN, GMM, or Canny.
        model         : String  Path to pre-trained model.
        device        : String  Device to be used for training and detecting. "cuda:0" or "cpu".
    """
    #
    if train_set is None and model is None:
        # regular bead data
        filename = os.path.join(os.getcwd(), "train")
        try:
            os.mkdir(filename)
        except:
            Logger.warning("Folder of regular beads already exists! Overwriting existing data.")

        bead_data_to_file(filename)
        train_set = [filename]

        # style bead data
        filename = os.path.join(os.getcwd(), "train_style")
        train_set.append(filename)
        sdset = StyleDatasetGen(dset=dset, len=100)
        sdset.gen_dataset()

    # Object detection
    if model is None:
        model = DNN(device=device)
        for d in train_set:
            print("Train set: " + d)
            d = BeadDatasetFile(d)
            train_dataloader = DataLoader(d, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
            model.train(train_dataloader)

        # save model
        model.save_model()
    else:
        model = DNN(model, device=device)

    # Tracking
    # Make directory
    kalman_dir = os.path.join(imgOutDir, "kalman")
    try:
        os.mkdir(kalman_dir)
    except:
        print("Folder already exists, overwriting contents ... ")

    Logger.detail("Detecting particles ...")
    
    for i in range(dset.length()-1):
        img = dset.get_img(i)
        bbox, mask = model.predict(img)
        
        # optical flow
        mask = mask.astype(np.bool)
        if len(img.shape)==3:
            cur = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            nxt = cv.cvtColor(dset.get_img(i+1),cv.COLOR_BGR2GRAY)
        else:
            cur = img
            nxt = dset.get_img(i+1)
        flow = cv.calcOpticalFlowFarneback(cur,nxt, None, pyr_scale=0.5, levels=5, 
                                           winsize=15, iterations=3, poly_n=5,
                                           poly_sigma=1.2, flags=0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        
        # Merge
        npar = len(bbox)
        
        # Average speed for each particle
        # cen   = np.zeros((npar,2)) # center position
        speed = np.zeros((npar,))  # average speed around the center
        for j in range(npar):
            # cen[j,:] = np.array([bbox[j,0] + bbox[j,2],          # x
            #                      bbox[j,1] + bbox[j,3]]) / 2.    # y
            # indxs = (xx-cen[j,0])**2 + (yy-cen[j,1])**2 <= r*r
            # indxs = np.logical_and(indxs, mask[j,...])
            speed[j] = np.mean(mag[mask[j,0,:,:]])
        
        # normalize speeds (useful for plotting later)
        max_speed = np.max(speed)
        speed     = speed/max_speed
        th_speed=0.2
        th_dist=2
        it =2
        mask, bbox, _ = mergeBoxes(mask, bbox, speed, mag, max_speed, th_speed, th_dist, it)
        
        # Draw bounding boxes
        cont = drawBox(img.copy(), bbox)

        # Show final image
        #cv.imshow("Frame", cont)
        cv.imwrite("{:s}/{:s}_{:d}.jpg".format(imgOutDir, modelType, i), cont)
        
        # Kalman tracking
        if i == 0:
            mot = MOT(bbox, mask)
        else:
            mot.step(bbox, mask)

        img_kalman = drawBlobs(img.copy(), mot.blobs)
        cv.imwrite("{:s}/{:s}_{:d}.jpg".format(kalman_dir, modelType, i), img_kalman)

        writeBlobs(mot.blobs, blobsOutFile, mot.cnt)