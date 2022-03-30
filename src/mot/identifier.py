import cv2 as cv
from src.mot.utils import drawBox, drawBlobs, writeBlobs
from src.mot.kalman import MOT
from src.mot.detectors import DNN
from src.datagen.bead_gen import bead_data_to_file, BeadDatasetFile, collate_fn
from src.datagen.style_data_gen_mask import StyleDatasetGen
import os
from src.logger import Logger
from torch.utils.data import DataLoader


def identify(dset, imgOutDir, blobsOutFile, model=None, train_set=None, crop=(512, 512)):
    """
    Identify particles using specified model.

    Attributes:
        fname         : String  Path to the video
        model         : String  Path to the DNN weights and config file
        imgOutDir     : String  Output folder of images with bounding boxes.
        blobsOutFile  : String  Output file for info of each identified particle.
        crop          : (int, int) Cropping sizes in x and y dimension.
    """
    #
    if train_set is None:
        # regular bead data
        filename = os.path.join(os.getcwd(), "train")
        try:
            os.mkdir(filename)
        except:
            print("Folder already exists")

        bead_data_to_file(filename)
        train_set = [filename]

        # style bead data
        filename = os.path.join(os.getcwd(), "train_style")
        train_set.append(filename)
        sdset = StyleDatasetGen(dset=dset, len=100)
        sdset.gen_dataset()


    # Object detection and kalman
    if model is None:
        model = DNN()
        for d in train_set:
            d = BeadDatasetFile(d)
            train_dataloader = DataLoader(d, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=4)
            model.train(train_dataloader)

    # Make directory
    try:
        os.mkdir(imgOutDir+"/kalman")
    except:
        print("Folder already exists, overwriting contents ... ")

    Logger.detail("Detecting particles ...")
    for i in range(dset.length()):
        img = dset.get_img(i)
        bbox, mask = model.predict(img)

        # Draw bounding boxes
        cont = drawBox(img.copy(), bbox)

        # Show final image
        # cv.imshow("Frame", cont)
        cv.imwrite("{:s}/dnn_{:d}.jpg".format(imgOutDir, i), cont)

        # Kalman tracking
        if i == 0:
            mot = MOT(bbox)
        else:
            mot.step(bbox)

        img_kalman = drawBlobs(img.copy(), mot.blobs)
        cv.imwrite("{:s}/kalman/dnn_{:d}.jpg".format(imgOutDir, i), img_kalman)

        writeBlobs(mot.blobs, blobsOutFile, mot.cnt)
