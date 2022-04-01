import cv2 as cv
from src.mot.utils import drawBox, drawBlobs, writeBlobs
from src.mot.kalman import MOT
from src.mot.detectors import DNN
import os
from src.logger import Logger

def identify(dset, imgOutDir, blobsOutFile, modelType = "DNN", model=None, train_set=None, gpu=True, crop=(512, 512)):
    """
    Identify particles using specified model.

    Attributes:
        fname         : String  Path to the video
        imgOutDir     : String  Output folder of images with bounding boxes.
        blobsOutFile  : String  Output file for info of each identified particle.
        modelType     : String  Type of detection model: DNN, GMM, or Canny.
        model         : String  Path to the DNN weights and config file
        crop          : (int, int) Cropping sizes in x and y dimension.
    """
    # Object detection and kalman
    #dset = Dataset(video_name=fname, crop=crop)
    detector = None
    if modelType.lower() == "dnn":
        detector  = DNN(dset=dset, fname=model, train_set=train_set)
    # TODO: Add back gmm and canny.
    elif modelType.lower() == "gmm":
        pass
    elif modelType.lower() == "canny":
        pass

    # Make directory
    try:
        os.mkdir(imgOutDir+"/kalman")
    except:
        print("Folder already exists, overwriting contents ... ")

    Logger.detail("Detecting particles ...")
    for i in range(dset.length()):
        img = dset.get_img(i)
        bbox, mask = detector.predict(i)
        
        # Draw bounding boxes
        cont = drawBox(img.copy(), bbox)

        # Show final image
        #cv.imshow("Frame", cont)
        cv.imwrite("{:s}/{:s}_{:d}.jpg".format(imgOutDir, modelType, i), cont)
        
        # Kalman tracking
        if i == 0:
            mot = MOT(bbox)
        else:
            mot.step(bbox)

        img_kalman = drawBlobs(img.copy(), mot.blobs)
        cv.imwrite("{:s}/kalman/dnn_{:d}.jpg".format(imgOutDir, i), img_kalman)

        writeBlobs(mot.blobs, blobsOutFile, mot.cnt)