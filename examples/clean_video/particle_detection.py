import os
from xmot.mot.identifier import identify
from xmot.dataset import Dataset
from xmot.logger import Logger

#----------------------------------------Detection------------------------------------------------#
# Input parameters
video = "./test_trim.mp4"                                   # Path to video.
output_dir = "./"                                           # Path to output folder.
model = "./model.pth"                                       # Path to model file.
                                                            # Comment out this line if wanting to train a new model.

# Initiate
blobsFile = os.path.join(output_dir, "blobs.txt")           # Output file
detection_img_dir = os.path.join(output_dir, "detection")   # Folder for output images
os.makedirs(detection_img_dir, exist_ok=True)

# Run the detection.
Logger.basic("Identifying particles in video: {:s}".format(video))
dset = Dataset(video, crop=(0, 0, 512, 512))
# For computer without GPU, use "device = 'cpu'"
identify(dset, detection_img_dir, blobsFile, model = model, device="cuda:0") 