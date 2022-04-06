# -*- coding: utf-8 -*-
"""
various bounding box detection schemes
"""
import os
import numpy as np
import cv2 as cv
import pickle

from src.logger import Logger
# from src.mot.utils import mergeBoxes
# from src.datagen.bead_gen import BeadDataset
# from src.datagen.style_data_gen_mask import StyleDataset

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms


class DNN:
    def __init__(self, model=None, num_classes=2, hidden_layer=256, device="cuda:0",
                 optimizer=None, lr=5e-3, momentum=0.9, weight_decay=5e-4,
                 lr_scheduler=None, step_size=3, gamma=0.1, nms_threshold=0.1, score_threshold=0.3):
        self.device = device
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        # DNN model
        if model is None:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            # bounding box regression
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # mask regression
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        else:
            self.model = model
        self.model.to(self.device)

        # Optimizer
        if optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer

        # scheduler
        if lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            self.lr_scheduler = lr_scheduler

        # for testing
        self.tsf = transforms.Compose([
            transforms.ToTensor()
        ])

    def _train_one_epoch(self, train_dataloader, print_interval):
        for images, targets in train_dataloader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

            if print_interval != 0:
                print(f'Loss summary: ')
                for l in loss_dict:
                    print(f'{l} {loss_dict[l]}')

            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

    def train(self, train_dataloader, epoch=20, print_interval=1000):
        self.model.train()
        for i in range(epoch):
            self._train_one_epoch(train_dataloader, print_interval)
        self.model.eval()

    def predict(self, img):
        img = self.tsf(img)
        with torch.no_grad():
            prediction = self.model([img.to(self.device)])
            bbox = prediction[0]["boxes"]
            mask = prediction[0]["masks"]
            scor = prediction[0]["scores"]

            # threshold boxes with small confidence scores
            indx = torch.where(scor>self.score_threshold)[0]
            bbox = bbox[indx, ...]
            mask = mask[indx, ...]
            scor = scor[indx, ...]

            # non-max suppression
            indx = torchvision.ops.nms(bbox, scor, self.nms_threshold)

        bbox = bbox[indx,...].to("cpu").data.numpy()
        mask = mask[indx,...].to("cpu").data.numpy()
        return bbox, mask


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

        return bbox, None


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
        return bbox, None

    def __train(self):
        while (True):
            _, img = self.cap.read()
            if img is None: break
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = gray[0:self.crop[0], 0:self.crop[1]]
            self.gmm.apply(gray)
        self.cap.release()
