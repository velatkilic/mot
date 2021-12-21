import cv2 as cv
from src.mot.utils import drawBox, drawBlobs, writeBlobs
from src.mot.kalman import MOT
from src.mot.detectors import DNN
from src.dataset import Dataset
from src.logger import Logger


def identify(fname, modeltype, imgOutDir, blobsOutFile, paramDir="data", gpu=True, crop=(512, 512)):
    """
    Identify particles using specified model.

    Attributes:
        fname         : String  Path to the video
        modeltype     : String  Path to the DNN weights
        imgOutDir     : String  Output folder of images with bounding boxes.
        blobsOutFile  : String  Output file for info of each identified particle.
        crop          : (int, int) Cropping sizes in x and y dimension.
    """
    # Object detection and kalman
    dset = Dataset(video_name=fname, crop=crop)

    detector = None
    if modeltype.lower() == "dnn":
        detector  = DNN(dset=dset, fname=paramDir, gpu=gpu)

    Logger.detail("Detecting particles ...")
    for i in range(dset.length()):
        img = dset.get_img(i)
        bbox, mask = detector.predict(i)
        
        # Draw bounding boxes
        cont = drawBox(img.copy(), bbox)
        
        # Show final image
        #cv.imshow("Frame", cont)
        cv.imwrite("{:s}/{:s}_{:d}.jpg".format(imgOutDir, modeltype, i), cont)
        
        # Kalman tracking
        if i == 0:
            mot = MOT(bbox)
        else:
            mot.step(bbox)
        
        img_kalman = drawBlobs(img.copy(), mot.blobs)
        writeBlobs(mot.blobs, blobsOutFile, mot.cnt)