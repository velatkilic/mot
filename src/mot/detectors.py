# -*- coding: utf-8 -*-
"""
various bounding box detection schemes
"""
import os
import numpy as np
import cv2 as cv
import pickle

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from src.logger import Logger
from src.mot.utils import mergeBoxes
from src.datagen.bead_gen import BeadDataset
from src.datagen.style_data_gen_mask import StyleDataset


class DNN:
    def __init__(self, fname=None, dset=None, th_speed=0.2, th_dist=2):
        # optic flow merge params
        self.th_speed = th_speed
        self.th_dist = th_dist

        # Dataset
        if dset is None:
            Logger.error("Dataset cannot be empty")
        self.dset = dset

        # DNN
        if fname is None:
            print("Entering DNN training mode ... ")
            self.__train()
            print("Training complete.")
        else:
            # Config
            with open(fname + "cfg.pkl", "rb") as file:
                cfg = pickle.load(file)
            cfg.merge_from_list(["MODEL.WEIGHTS", fname + "model_final.pth"])
            self.predictor = DefaultPredictor(cfg)  # create predictor from the config

    def predict(self, idx):
        # get DNN predictions
        img = self.dset.get_img(idx)

        # convert gray since predictor expects BGR
        if self.dset.gray:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        out = self.predictor(img)
        bbox = out["instances"].pred_boxes.to('cpu').tensor.data.numpy()
        mask = out["instances"].pred_masks.to('cpu').data.numpy().astype(np.bool)

        # optical flow fuse
        if (idx + 1) < self.dset.length():
            gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to gray scale
            nxt = self.dset.get_img(idx + 1)  # get next frame
            if not self.dset.gray:
                nxt = cv.cvtColor(nxt, cv.COLOR_BGR2GRAY)  # convert to gray scale
            flow = cv.calcOpticalFlowFarneback(gry, nxt, None, pyr_scale=0.5, levels=5,
                                               winsize=15, iterations=3, poly_n=5,
                                               poly_sigma=1.2, flags=0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

            npar = len(bbox)
            speed = np.zeros((npar,))  # average speed around the center
            for j in range(npar):
                speed[j] = np.mean(mag[mask[j, :, :]])
            # normalize speeds (useful for plotting later)
            max_speed = np.max(speed)
            speed = speed / max_speed

            # Fuse if speed is similar
            mask, bbox, _ = mergeBoxes(mask, bbox, speed, mag,
                                       max_speed, self.th_speed, self.th_dist, 2)

        return bbox, mask

    def __get_bead_dicts(self, img_dir):
        return np.load(img_dir + '/annot.npz', allow_pickle=True)['dataset_dicts']

    def __train(self):
        # Create bead dataset
        print("Creating bead dataset without style transfer")
        bdset = BeadDataset()
        bdset.gen_dataset()
        print("Bead dataset generation without style transfer complete.")

        # Create bead dataset with style transfer
        print("Creating bead dataset with style transfer")
        sdset = StyleDataset(dset=self.dset)
        sdset.gen_dataset()
        print("Bead dataset generation with style transfer complete.")

        # register training dataset
        for d in ["train", "train_style"]:
            DatasetCatalog.register("beads_" + d, lambda d=d: self.__get_bead_dicts(os.getcwd() + "/" + d))
            MetadataCatalog.get("beads_" + d).set(thing_classes=["particle"])

        Logger.basic("Training DNN ... ")
        # Config for training
        cfg = get_cfg()
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("beads_train", "beads_train_style")
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # learning rate
        cfg.SOLVER.MAX_ITER = 300  # number of iterations
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # region of interest (ROI) head batchsize
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only particle class

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)  # init trainer
        trainer.resume_or_load(resume=False)
        trainer.train()  # train

        # cfg already contains everything we've set previously. Now we changed it a little bit for inference:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set a custom testing threshold, (smaller leads to more detections)
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # non-max suppression threshold
        self.predictor = DefaultPredictor(cfg)  # create predictor from the config
        print("DNN training complete.")


class Canny:
    def __init__(self, th1=50, th2=100, minArea=20, it_closing=1):
        self.th1 = th1
        self.th2 = th2
        self.minArea = minArea
        self.it_closing = it_closing

    def predict(self, img):
        # convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Canny edge detection
        edge = cv.Canny(gray, self.th1, self.th2)

        # Morphological transformation: closing
        kernel = np.ones((8, 8), dtype=np.uint8)
        closing = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel, iterations=self.it_closing)

        # Contours
        contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Bounding rectangle
        bbox = []
        for temp in contours:
            area = cv.contourArea(temp)
            if area > self.minArea:
                x, y, w, h = cv.boundingRect(temp)
                bbox.append([x, y, x + w, y + h])

        return bbox


class GMM:
    def __init__(self, fname, crop, history=100, varThreshold=40, it_closing=1, minArea=20):
        """

        Attributes:
            fanme        : String     Path to the video
            crop         : (int, int) Sizes in x and y dimension for cropping each video frame. 
            history      : int        Length of history 
            varThreshold : int        Threshold of pixel background identification in MOB2 of cv.
            it_closing   : int        Parameter of cv.morphologyEx
            minArea      : int        Minimal area of bbox to be considered as a valid particle.
        """
        self.cap = cv.VideoCapture(fname)
        self.it_closing = it_closing
        self.minArea = minArea
        self.crop = crop
        self.gmm = cv.createBackgroundSubtractorMOG2(history=history,
                                                     varThreshold=varThreshold)
        Logger.detail("Training GMM ...")
        self.__train()

    def predict(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask = self.gmm.apply(gray)
        # Morphological transformation: closing
        kernel = np.ones((8, 8), dtype=np.uint8)
        closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=self.it_closing)

        # Contours
        contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Bounding rectangle
        bbox = []
        for temp in contours:
            area = cv.contourArea(temp)
            if area > self.minArea:
                x, y, w, h = cv.boundingRect(temp)
                bbox.append([x, y, x + w, y + h])
        return bbox

    def __train(self):
        while (True):
            _, img = self.cap.read()
            if img is None: break
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = gray[0:self.crop[0], 0:self.crop[1]]
            self.gmm.apply(gray)
        self.cap.release()
