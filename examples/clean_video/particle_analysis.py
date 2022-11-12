import os
import cv2 as cv
import numpy as np

from xmot.digraph.digraph import Digraph
from xmot.digraph.utils import load_blobs_from_text, collect_images, paste_images, generate_video
from xmot.digraph import commons
from xmot.logger import Logger

# Input Parameters
output_dir = "./"                     # Path to folder to write output.
blobsFile = "./blobs.txt"             # Path to the output file from particle_detection.py
video = "./test_trim.mp4"             # Path to the video
detection_img_dir = os.path.join(output_dir, "detection") # Path to the "detection" folder from particle_detection.py

# Initiate parameters
reproduce_img_dir = os.path.join(output_dir, "reproduced")
merged_img_dir = os.path.join(output_dir, "merged")
os.makedirs(reproduce_img_dir, exist_ok=True)
os.makedirs(merged_img_dir, exist_ok=True)
reproduced_video = os.path.join(output_dir, "reproduced.avi")

# Load identified particles into digraph
particles = load_blobs_from_text(blobsFile)
dg = Digraph()
commons.PIC_DIMENSION = [512, 512]
dg.add_video(particles) # Load particles identified in the video.
dg.detect_particle_shapes(video)

# Write to terminal the detailed information of the digraph.
Logger.basic("Detailed information of digraph: \n" + str(dg))

Logger.basic("Drawing reproduced images ...")
draw_id = True
write_meta = True
draw_shape = True

# Generate reproduced video in various format
rep_imgs = dg.draw(reproduce_img_dir, write_meta, draw_id, draw_shape)
#rep_imgs = dg.draw_overlay(reproduce_img_dir, write_meta, draw_id)
#rep_imgs = dg.draw_line_format(reproduce_img_dir, write_meta, draw_id)

Logger.basic("Reproducing video ...")
cap = cv.VideoCapture(video)
num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
cap.release()
orig_imgs = collect_images(detection_img_dir, "{:s}_".format("DNN"), "jpg", 0, num_frames - 1)
merged_imgs = paste_images(orig_imgs, rep_imgs, merged_img_dir, write_meta)

cv_imgs = [cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR) for img in merged_imgs]
generate_video(cv_imgs, reproduced_video)