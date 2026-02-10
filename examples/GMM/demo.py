from pathlib import Path

from xmot.mot.detectors import GMM
from xmot.utils.image_utils import load_images_from_dir
from xmot.utils.benchmark_utils import save_prediction_bbox, save_prediction_cnt
from xmot.mot.identifier import build_trajectory_batch_GMM
from xmot.digraph.digraph import Digraph
from xmot.digraph.parser import load_blobs_from_text
from xmot.digraph import commons

#### Config
# Use the default GMM training strategy, i.e. the "Half Video" strategy in paper.
history = -1
distance = -1

#### Input and Output. You can change to your own paths.
orig_images_dir = "xmot_example_data/frames_orig"  # Point to the submodule folder.
bf_images_dir = "xmot_example_data/frames_brightfield_subtracted" # Point to the submodule folder.
outdir = "./output"
output_name = "./digraph.txt"
debug_image_interval = 25  # how often a debug image should be drawn.

outdir_path = Path(outdir)
outdir_path.mkdir(exist_ok=True)
blob_file = str(outdir_path.joinpath("blobs.txt"))
kalman_dir = str(outdir_path.joinpath("kalman"))



#=================================== The main script ===================================#
#### Load images
images_orig, orig_file_names = load_images_from_dir(orig_images_dir)
images_bf_subtracted, bf_file_names = load_images_from_dir(bf_images_dir)
commons.PIC_DIMENSION = list(reversed(images_bf_subtracted[0].shape)) # [width, height]: columns, rows
outId = list(range(0, len(images_bf_subtracted), debug_image_interval))  # A subset of frames to be drawn for debugging purposes.

#### Step 1: Detecting particles from each frame
gmm = GMM(images=images_bf_subtracted,
          train_images=images_bf_subtracted,
          orig_images=images_orig)
dict_bbox, dict_cnt = gmm.predict_by_batch(history=history,
                                           distance=distance,
                                           outdir=str(outdir_path),
                                           outId=outId)

# Save detection results for debugging purposes
save_prediction_bbox(str(outdir_path.joinpath("gmm_bbox.txt")), dict_bbox) # For debug
save_prediction_cnt(str(outdir_path.joinpath("gmm_cnt.npy")), dict_cnt)    # For debug

#### Step 2: Kalman Filter and Hungarian algorithm to build trajectories frame by frame
build_trajectory_batch_GMM(dict_bbox, dict_cnt, images_bf_subtracted, kalman_dir, blobs_out=blob_file)

#### Step 3: Build graph and shape analysis:
particles = load_blobs_from_text(blob_file)

print("Constructing digraph ...")
dg = Digraph()
dg.add_video(particles) # Load particles identified in the video.
dg.detect_particle_shapes(images=images_bf_subtracted)

print("Drawing events ...")
dg.draw_events(images_orig, outdir=str(outdir_path.joinpath("events")), prefix ="example")

# Write to terminal the detailed information of the digraph/video.
# Redirect the output to a file if want to save it.
with open(outdir_path.joinpath(output_name), 'w') as f:
    f.write(str(dg))
