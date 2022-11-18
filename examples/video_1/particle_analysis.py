import os
import cv2 as cv
import numpy as np

from xmot.digraph.digraph import Digraph
from xmot.digraph.utils import collect_images, paste_images, generate_video
from xmot.digraph.parser import load_blobs_from_text
from xmot.digraph import commons
from xmot.logger import Logger

# User Input Parameters
output_dir = "./"                     # Path to folder to write output.
blobsFile = "./blobs.txt"             # Path to the output file from particle_detection.py
video = "./example.mp4"               # Path to the video
detection_img_dir = os.path.join(output_dir, "detection") # Path to the "detection" folder from particle_detection.py
commons.PIC_DIMENSION = [512, 512]    # Crop of the video. Delete this line if no cropping is wanted.

# Initiate parameters
reproduce_img_dir = os.path.join(output_dir, "reproduced")
merged_img_dir = os.path.join(output_dir, "merged")
os.makedirs(reproduce_img_dir, exist_ok=True)
os.makedirs(merged_img_dir, exist_ok=True)
reproduced_video = os.path.join(output_dir, "reproduced.avi")
cap = cv.VideoCapture(video)
num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
if commons.PIC_DIMENSION[0] == 0 or commons.PIC_DIMENSION[1] == 0:
    commons.PIC_DIMENSION = [int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))]
cap.release()

# Load identified particles into digraph
particles = load_blobs_from_text(blobsFile)
dg = Digraph()
dg.add_video(particles) # Load particles identified in the video.
dg.detect_particle_shapes(video)

# Write to terminal the detailed information of the digraph.
Logger.basic("Detailed information of digraph: \n" + str(dg))

Logger.basic("Drawing reproduced images ...")
draw_id = True
write_meta = True
draw_shape = True

# Generate reproduced video in various format
# Give num_frames to make sure having equal number of reproduced images as the original video images.
rep_imgs = dg.draw(reproduce_img_dir, write_meta, draw_id, draw_shape, start_frame=0, end_frame=num_frames)
#rep_imgs = dg.draw_overlay(reproduce_img_dir, write_meta, draw_id)
#rep_imgs = dg.draw_line_format(reproduce_img_dir, write_meta, draw_id)

Logger.basic("Reproducing video ...")
orig_imgs = collect_images(detection_img_dir, "{:s}_".format("DNN"), "jpg", 0, num_frames - 1)
merged_imgs = paste_images(orig_imgs, rep_imgs, merged_img_dir, write_meta)

cv_imgs = [cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR) for img in merged_imgs]
generate_video(cv_imgs, reproduced_video)