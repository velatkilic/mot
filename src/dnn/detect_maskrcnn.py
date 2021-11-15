#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified from detectron tutorial:
    https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html
"""

# import some common libraries
import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mname', type=str, default='test', help='MP4 movie name')

args = parser.parse_args()

import imageio

cwd = os.getcwd()  # Get current working directory
mname = cwd + '/' + args.mname + '.mp4'  # Movie name
vid = imageio.get_reader(mname, 'ffmpeg')  # Read video

leng = vid.count_frames()  # count number of frames


# get frame idx from video
def get_img(idx):
    img = vid.get_data(idx)
    return img[0:512, 0:512]


# load bead dictionary from a given directory
def get_bead_dicts(img_dir):
    return np.load(img_dir + '/annot.npz', allow_pickle=True)['dataset_dicts']


# register trainin dataset
for d in ["train"]:
    DatasetCatalog.register("beads_" + d, lambda d=d: get_bead_dicts(cwd + "/" + d))
    MetadataCatalog.get("beads_" + d).set(thing_classes=["particle"])
# bead_metadata = MetadataCatalog.get("bead_train")

# Config for training
cfg = get_cfg()
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("beads_train",)
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
predictor = DefaultPredictor(cfg)  # create predictor from the config

try:
    os.mkdir(cwd + "/video")
except:
    print("Video folder already exits")

try:
    os.mkdir(cwd + "/video/boxes")
except:
    print("Boxes folder already exists")

import scipy.io as sio

leng = vid.count_frames()
bboxes = []
masks = []
for i in range(leng):
    im = get_img(i)
    outputs = predictor(
        im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    # bboxes.append(outputs["instances"].pred_boxes.to('cpu').tensor.data.numpy())
    bbox = outputs["instances"].pred_boxes.to('cpu').tensor.data.numpy()
    mask = outputs["instances"].pred_masks.to('cpu').data.numpy().astype(np.bool)
    sio.savemat(cwd + '/video/boxes/frame' + str(i) + '.mat', {'bbox': bbox, 'mask': mask})

    bboxes.append(bbox)
    masks.append(mask)

    v = Visualizer(im[:, :, ::-1], scale=1.)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    im_out = out.get_image()[:, :, ::-1]
    cv2.imwrite(cwd + '/video/frame' + str(i) + '.png', im_out)

import pickle

with open(cwd + "/video/boxes/bboxes.txt", "wb") as fp:  # Pickling
    pickle.dump(bboxes, fp)

with open(cwd + "/video/boxes/masks.txt", "wb") as fp:  # Pickling
    pickle.dump(masks, fp)