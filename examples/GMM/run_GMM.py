import os
from pathlib import Path

from xmot.mot.detectors import GMM
from xmot.utils.image_utils import load_images_from_dir
from xmot.utils.benchmark_utils import save_prediction_bbox, save_prediction_cnt
from xmot.utils.benchmark_utils import load_prediction_bbox, load_prediction_cnt
from xmot.mot.identifier import build_trajectory_batch_GMM
from xmot.digraph.digraph import Digraph
from xmot.digraph.utils import collect_images, paste_images, generate_video
from xmot.digraph.parser import load_blobs_from_text
from xmot.digraph import commons
from xmot.logger import Logger

#### Config
AREA_THRESHOLD = 100

#### Inputs
input_video_frames_dir = "/path/to/folder/of/your/frames"
input_video_frames_dir = "/path/to/folder/of/your/background_subtracted/frames" # To subtract background, see
history = -1
distance = 1
outdir = "."
blob_file = "./blobs.txt"
kalman_dir = "./kalman"

#### Load Images
input_path = Path(input_video_dir)
outdir_path = Path(outdir)
outdir_path.mkdir(exist_ok=True)

images_orig, orig_file_names = load_images_from_dir(str(input_path.joinpath("frames")))
images_bf_subtracted, bf_file_names = load_images_from_dir(str(input_path.joinpath("frames_brightfield_subtracted")))
commons.PIC_DIMENSION = list(reversed(images_bf_subtracted[0].shape)) # [width, height]
outId = list(range(0, len(images_bf_subtracted), 50))  # A subset of frames to draw out for debugging.


#### Step 1: Detecting particles from each frame
gmm = GMM(images=images_bf_subtracted,
          train_images=images_bf_subtracted, 
          orig_images=images_orig, 
          area_threshold=AREA_THRESHOLD)
dict_bbox, dict_cnt = gmm.predict_by_batch(history=history,
                                           distance=distance,
                                           outdir=str(outdir_path),
                                           outId=outId)

# See bload_prediction
save_prediction_bbox(str(outdir_path.joinpath("gmm_bbox.txt")), dict_bbox) # For debug
save_prediction_cnt(str(outdir_path.joinpath("gmm_cnt.npy")), dict_cnt)    # For debug

#### Step 2: Kalman Filter implementation frame by frame
build_trajectory_batch_GMM(dict_bbox, dict_cnt, images_bf_subtracted, kalman_dir, blobs_out=blob_file)

#### Step 3: Build graph and shape analysis:
particles = load_blobs_from_text(blob_file)
dg = Digraph()
dg.add_video(particles) # Load particles identified in the video.
dg.detect_particle_shapes(images=images_bf_subtracted)

# Write to terminal the detailed information of the digraph/video.
# Redirect the output to a file if want to save it.
# TODO: We still need a better and more streamlined way to extract properties from the data
# structure rather than plainly printing it.
Logger.basic("Detailed information of digraph: \n" + str(dg))
