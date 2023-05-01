# -*- coding: utf-8 -*-
"""
various bounding box detection schemes
"""
import os
import numpy as np
import cv2 as cv
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

from xmot.logger import Logger
from xmot.config import AREA_THRESHOLD
from xmot.mot.utils import drawBox
from xmot.utils.image_utils import get_contour_center
# from xmot.mot.utils import mergeBoxes
# from xmot.datagen.bead_gen import BeadDataset
# from xmot.datagen.style_data_gen_mask import StyleDataset

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
class DNN:
    def __init__(self, model=None, num_classes=2, hidden_layer=256, device="cuda:0",
                 optimizer=None, lr=5e-3, momentum=0.9, weight_decay=5e-4,
                 lr_scheduler=None, step_size=3, gamma=0.1, nms_threshold=0.1, score_threshold=0.3): # nms_threshold=0.1, score_threshold=0.3
        self.device = torch.device(device)
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
            self.model = torch.load(model, map_location=self.device)

        self.model.to(device)

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
        self.model.train() # Set on training mode, not actually updating the parameter.
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
            if self.score_threshold is not None:
                indx = torch.where(scor > self.score_threshold)[0]
                bbox = bbox[indx, ...]
                mask = mask[indx, ...]
                scor = scor[indx, ...]

            # non-max suppression
            if self.nms_threshold is not None:
                indx = torchvision.ops.nms(bbox, scor, self.nms_threshold)
                bbox = bbox[indx, ...]
                mask = mask[indx, ...]
                scor = scor[indx, ...]

        bbox = bbox.to("cpu").data.numpy()
        mask = mask.to("cpu").data.numpy()
        scor = scor.to("cpu").data.numpy()
        return bbox, mask#, scor

    def save_model(self, path="model.pth"):
        torch.save(self.model, path)


class Canny:
    """
    Obsolete
    """
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
    """
    GMM to separate background with foreground moving objects. The foreground objects will
    be the detected particles.

    In contrary to the DNN and binary thresholding method, GMM only works for a video
    in its entirity, rather than for individual frames or segments of the video.
    """
    
    KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    CNT_THICKNESS = 1 # Thickness of contour when drawing.
    
    def __init__(self, images: List[np.ndarray], train_images: List[np.ndarray] = None, crop: List[int] = None, 
                 history=-1, varThreshold=16, it_closing=1, area_threshold=AREA_THRESHOLD):
        """
        Attributes:
            images         : List       List of frames.
            train_images   : List       List of frames used for training. By default, it's the same
                                        as the images to be detected.
            crop           : (int, int) Sizes in x and y dimension for cropping each video frame. 
            history        : int        Length of history. Use same Default as OpenCV.
            varThreshold   : int        Threshold of pixel background identification in MOB2 of cv.
                                        Use same default as OpenCV.
            it_closing     : int        Parameter of cv.morphologyEx
            area_threshold : int        Minimal area of bbox to be considered as a valid particle.
        """
        self.images = images
        self.train_images = train_images if train_images != None else None
        self.it_closing = it_closing
        self.area_threshold = area_threshold
        self.crop = crop
        #if history == -1:
        #    history = int(len(images) / 2) # Default history separate 
        #self.gmm = cv.createBackgroundSubtractorMOG2(history=history,
        #                                             varThreshold=varThreshold)
        #self.__train()

    def predict_by_batch(self, history=-1, distance=1, mahal_distance=16.0, outdir=None) \
        -> Tuple[Dict[int, List[int]], Dict[int, List[np.ndarray]]]:
        """
        Train the background using a batch of frames in the future and then detect the current
        batch of frames. It separate the images under detection with the background as far as
        possible, to prevent slowly moving objects to be completely shadowed by the background
        detected in the GMM model.

        There're several strategies to train background:
        1. "distance=0, history < len(self.images), self.images == self.train_images":
           Train new background batch by batch, but use the images under detection as the training
           images.
        2. "distance>0, history < len(self.images), self.images == self.train_images":
           Train new background batch by batch. Use frames in a future batch to train background
           for detection of the current batch.
        3. "distance=0, history = len(self.images), self.images == self.train_images":
           Use the entire video to train background first and do not update the background anymore.
        4. "distance=0, history = len(self.train_images), self.train_images = brightfield images":
           Use the whole set of brightfield images to train background first and 
           do not update the background anymore.

        Attributes:
            mahal_distance : double   Threshold on the squared Mahalanobis distance
                                      between a pixel and the mixture of guassians to 
                                      decide whether a pixel belongs to the background 
                                      (well described) or not.
            outdir         : str      When not None, write full set of intermediate images to the
                                      outdir. Primarily for debugging purposes.
        Return:
            1. dict of bboxes in each image, with frame id as key.
            2. dict of contours in each images, with frame id as key. The contours in each list
               match the order of corresponding bbox in the lists of the first dict.
        """
        if history == -1:
            history = int(len(self.images) / 2)
            distance = 1

        if outdir != None:
            outDir = Path(outdir)
            backgroundDir = outDir.joinpath("background")
            backgroundDir.mkdir(exist_ok=True)
            plainForegroundDir = outDir.joinpath("plain_foreground")
            plainForegroundDir.mkdir(exist_ok=True)
            processedForegroundDir = outDir.joinpath("processed_foreground")
            processedForegroundDir.mkdir(exist_ok=True)
            #contouredOnSubtractedDir = outDir.joinpath("contour_on_subtracted")
            #contouredOnSubtractedDir.mkdir(exist_ok=True)
            masksDir = outDir.joinpath("contour_as_masks")
            masksDir.mkdir(exist_ok=True)
            centroidDir = outDir.joinpath("centroid") # Plot centroid of the contours
            centroidDir.mkdir(exist_ok=True)

        gmm = cv.createBackgroundSubtractorMOG2(history=history,
                                                varThreshold=mahal_distance)
        dict_bbox = {}
        dict_contours = {}

        i_detect = 0
        i_train = i_detect + distance * history
        while(i_detect < len(self.images)):
            #gmm.apply(images[i_train % len(images)], 0, learningRate=1) # Reinitialize the history
            for i in range(i_train, i_train + history):
                gmm.apply(self.train_images[i % len(self.train_images)])
            for i in range(i_detect, min(i_detect + history, len(self.images))):
                orig_mask = gmm.apply(self.images[i], None, learningRate=0) # rate=0: stop updating the background.
            
                # Remove noise
                mask = cv.morphologyEx(orig_mask, cv.MORPH_OPEN, GMM.KERNEL, iterations=1)
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, GMM.KERNEL, iterations=1)

                # Close gaps
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, GMM.KERNEL, iterations=2)
                
                # Remove noise again
                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, GMM.KERNEL, iterations=1)
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, GMM.KERNEL, iterations=1)
                
                if outdir != None:
                    cv.imwrite(str(backgroundDir.joinpath(f"background_{i}.png")), gmm.getBackgroundImage())
                    cv.imwrite(str(plainForegroundDir.joinpath(f"plainForeground_{i}.png")), orig_mask)
                    cv.imwrite(str(processedForegroundDir.joinpath(f"processedForeground_{i}.png")), mask)

                # Since objects in the mask are white, add a thin black padding to the boundaries
                # of the whole image.
                img_padded = cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)

                contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                filtered_contours = [] # Contours that are at least the area_threshold
                bbox = []
                for cnt in contours:
                    area = cv.contourArea(cnt)
                    if area > self.area_threshold:
                        x, y, w, h = cv.boundingRect(cnt)
                        bbox.append([x, y, x + w, y + h])
                        filtered_contours.append(cnt)
                
                dict_bbox[i_detect] = bbox
                dict_contours[i_detect] = filtered_contours
                
                if outdir != None:
                    #img_contoured = cv.drawContours(cv.cvtColor(images[i], cv.COLOR_GRAY2BGR), 
                    #        filtered_contours, -1, (0, 0, 255), thickness=GMM.CNT_THICKNESS)
                    #cv.imwrite(str(contouredOnSubtractedDir.joinpath("GMM_{:d}.png".format(i))), img_contoured)
                    
                    orig_img_bbox = drawBox(cv.cvtColor(self.images[i], cv.COLOR_GRAY2BGR), bbox)
                    cv.imwrite(str(outDir.joinpath("GMM_{:d}.png".format(i))), orig_img_bbox)
                    
                    # Use contour as red Masks.
                    orig_img_contoured = cv.drawContours(cv.cvtColor(self.images[i], cv.COLOR_GRAY2BGR), 
                            filtered_contours, -1, (0, 0, 255), thickness=GMM.CNT_THICKNESS)
                    cv.imwrite(str(masksDir.joinpath("GMM_{:d}.png".format(i))), orig_img_contoured)
                    
                    # Plot centroid of contours out.
                    #orig_img_centroid = np.copy(orig_img_contoured)
                    for cnt in filtered_contours:
                        orig_img_centroid = cv.circle(orig_img_centroid, get_contour_center(cnt),
                                radius = 3, color=(0, 255, 0), thickness=-1)
                    cv.imwrite(str(centroidDir.joinpath("GMM_{:d}.png".format(i))), orig_img_centroid)

            i_train = i_train + history
            i_detect = min(i_detect + history, len(self.images))
        
        return dict_bbox, dict_contours

    def predict(self, img, learningRate=-1):
        """
        Deprecated.

        learningRate : int  Controls how background image is updated when gmm.apply() is called.
                            -1: automatic update background image;
                            0: don't update background image using the current image
                            1: completely reinitialize the background image based on the current image.
        """
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask = self.gmm.apply(gray, learningRate=learningRate)
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
