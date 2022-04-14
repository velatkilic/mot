import os
import click
import cv2 as cv
from PIL import Image
from cv2 import merge
import numpy as np
from typing import List

import sys
sys.path.append("/mnt/d/JHU/Research/Machine_Learning_Characterization/code/mot/")

from src.mot.identifier import identify
from src.dataset import Dataset
from src.digraph.digraph import Digraph
from src.digraph.utils import load_blobs_from_text, collect_images, paste_images, generate_video
from src.digraph import commons
from src.logger import Logger

#----------------------------------------Detection------------------------------------------------#
# Initiate parameters:
video = os.path.join("./", "test_trim.mp4")                 # Input video
output_dir = "./"
blobsFile = os.path.join(output_dir, "blobs.txt")           # Output file
detection_img_dir = os.path.join(output_dir, "detection")   # Folder for output images
os.makedirs(detection_img_dir, exist_ok=True)

Logger.basic("Identifying particles in video: {:s}".format(video))
dset = Dataset(video, crop=(0, 0, 512, 512))
identify(dset, detection_img_dir, blobsFile, model = "model.pth", device="cpu")

exit()













Logger.basic("Reading identified particles ...")
particles = load_blobs_from_text(blobsFile)

Logger.detail("Loading particles into digraph ...")
dg = Digraph()
commons.PIC_DIMENSION = [512, 512]
dg.add_video(particles) # Load particles identified in the video.

Logger.detail("Detailed information of digraph: \n" + str(dg))

Logger.basic("Drawing reproduced images ...")
draw_id = True
write_meta = True
#rep_imgs = dg.draw(reproduce_img_dir, write_meta, draw_id)
rep_imgs = dg.draw_overlay(reproduce_img_dir, write_meta, draw_id)
#rep_imgs = dg.draw_line_format(reproduce_img_dir, write_meta, draw_id)

Logger.basic("Reproducing video ...")
cap = cv.VideoCapture(video)
num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
cap.release()
orig_imgs = collect_images(detection_img_dir, "{:s}_".format("dnn"), "jpg", 0, num_frames - 1)
merged_imgs = paste_images(orig_imgs, rep_imgs, merged_img_dir, write_meta)

cv_imgs = [cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR) for img in merged_imgs]
generate_video(cv_imgs, reproduced_video)