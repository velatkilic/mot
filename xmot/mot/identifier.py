import cv2 as cv
from xmot.mot.utils import drawBox, drawBlobs, writeBlobs
from xmot.mot.kalman import MOT
from xmot.mot.detectors import DNN
from xmot.datagen.bead_gen import bead_data_to_file, BeadDatasetFile, collate_fn
from xmot.datagen.style_data_gen_mask import StyleDatasetGen
import os
from xmot.logger import Logger
from torch.utils.data import DataLoader

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
    for i in range(dset.length()):
        img = dset.get_img(i)
        bbox, mask = model.predict(img)
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