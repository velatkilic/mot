import os
from xmot.mot.identifier import identify
from xmot.dataset import Dataset
from xmot.logger import Logger
from xmot.digraph import commons
import cv2 as cv

# User input parameters
output_dir = "./"                     # Path to output folder.
video = "./example.mp4"             # Path to video.
model = "./model.pth"                 # Path to a pre-trained model file.
                                      # Use "model = None" if users want to train a new model.
commons.PIC_DIMENSION = [512, 512]    # Dimension of images to be cropped from the video.
                                      # Delete this line if no cropping is wanted.
device = "cuda:0"                     # "cuda:0" or "cpu". Whether use GPU or CPU in training and detection.

# Initiate
blobsFile = os.path.join(output_dir, "blobs.txt")           # Output file
detection_img_dir = os.path.join(output_dir, "detection")   # Folder for output images
os.makedirs(detection_img_dir, exist_ok=True)
cap = cv.VideoCapture(video)
num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
if commons.PIC_DIMENSION[0] == 0 or commons.PIC_DIMENSION[1] == 0:
    commons.PIC_DIMENSION = [int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))]
cap.release()

# Run the detection.
Logger.basic("Identifying particles in video: {:s}".format(video))
dset = Dataset(video, crop=(0, 0, commons.PIC_DIMENSION[0], commons.PIC_DIMENSION[1]))
# For computer without GPU, use "device = 'cpu'"
identify(dset, detection_img_dir, blobsFile, model = model, device=device) 