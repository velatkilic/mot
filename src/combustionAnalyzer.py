import os
import click

from mot.identifier import identify
from digraph.digraph import Digraph
from digraph.utils import load_text, collect_images, paste_images, generate_video
from digraph import commons
from cv2 import cv

from logger import Logger

@click.command()
@click.argument("video")
@click.argument("output_dir")
@click.option("-m", "--write-meta-data", "write_meta", is_flag=True, default=True,
              help="Whether write meta-data like bbox of identified particles "\
                   "and reproduced pictures from the diagraph.")
@click.option("-c", "--crop", default=512, type=(int, int),
              help="Crop sizes in x and y dimension for each video frame.")
@click.option("-d", "--draw-type", default="plain", type=str,
              help="Modes of pictures to reproduce: 'plain', 'overlay', 'line'."\
                   "'plain' means regular picture with particles drawn as circles filling bbox."\
                   "'overlay' means particles of all frames are drawn on the same picture."\
                   "'line' means particles are drawn in lines to emphasize trajectories.")
@click.option("-o", "--io", default="warning", type=str,
              help="IO level. Available options are: 'quiet', 'baisc' "\
                   "'warning', 'detail', 'debug'")
def combustionAnalyzer(video, output_dir, write_meta, crop, draw_type):
    """
    Identify particles from videos of combustion via neural networks. Then track
    and analyze trajectories of particles using Kalmen filter and digraph.

    Read VIDEO as input and write particle informations and reproduced video
    in OUTPUT_DIR/blobs.txt and OUTPUT_DIR/reproduced.avi. Additional meta-data
    produced during the process will be wrote in OUTPUT_DIR/detection
    and OUTPUT_DIR/reproduced.

    \b
    Properties of videos being analyzed include:
    * Particle size distribution (spherical/molten droplets only)
    * Particle velocity vs Particle size
    * Presence of agglomerates
    * Size distribution of agglomerates/clusters
    * Presence of bubbles/voids
    * Bubble size distribution
    * Number of bubbles per particle vs particle size
    * Bubble size vs Particle size
    * Bubble growth rate vs Particle Size
    * Frequency of micro-explosions
    * Micro-explosion vs particle size
    * Any other anomalies
    """
    # Outputs
    blobsFile = os.path.join(output_dir, "blobs.txt")
    reproduced_video = os.path.join(output_dir, "reproduced.avi")

    if os.path.exists(blobsFile):
        os.remove(blobsFile)
    
    # Optional Meta-data 
    detection_img_dir = os.path.join(output_dir, "detection")
    reproduce_img_dir = os.path.join(output_dir, "reproduced")
    merged_img_dir = os.path.join(output_dir, "merged")

    if not os.path.exists(detection_img_dir):
        os.makedirs(detection_img_dir)
    if not os.path.exists(reproduce_img_dir):
        os.makedirs(reproduce_img_dir)
    if not os.path.exists(merged_img_dir):
        os.makedirs(merged_img_dir)

    Logger.basic("Loading video ...")
    num_frames = count_frame(video)
    # We pass on the video path since each function needs its own video pointer.
    identify(video, detection_img_dir, blobsFile, crop=crop)

    Logger.basic("Reading identified particles ...")
    particles = load_text(blobsFile)

    Logger.detail("Loading particles into digraph ...")
    dg = Digraph()
    commons.PIC_DIMENSION = crop
    dg.add_video(particles) # Load particles identified in the video.

    Logger.basic("Drawing reproduced images ...")
    if draw_type.lower() == "plain":
        rep_imgs = dg.draw(reproduce_img_dir, write_meta)
    elif draw_type.lower() == "overlay":
        rep_imgs = dg.draw_overlay(reproduce_img_dir, write_meta)
    elif draw_type.lower() == "line":
        rep_imgs = dg.draw_line_format(reproduce_img_dir, write_meta)
    else:
        Logger.warning("Invalid drawing mode: " + draw_type + " Use plain.")
        rep_imgs = dg.draw(reproduce_img_dir, write_meta)

    Logger.basic("Reproducing video ...")
    orig_imgs = collect_images(detection_img_dir, "gmm_", "jpg", 0, num_frames - 1)
    #rep_imgs = collect_images(reproduce_img_dir, "reproduced_", "png", 0, num_frames - 1)
    merged_imgs = paste_images(rep_imgs, orig_imgs, merged_img_dir, write_meta)

    # <TODO> Find whether Pillow images work with opencv
    files = [f for f in os.listdir(merged_img_dir) if f.startswith("merged_") and f.endswith("png")]
    files.sort(key=lambda f: int(f.replace("merged_", "").replace("." + "png", "")))
    files = [os.path.join(merged_img_dir, f) for f in files]
    generate_video(files, reproduced_video)

if __name__ == "__main__":
    combustionAnalyzer()

def count_frame(video) -> int:
    cap = cv.CaptureVideo(video)
    num_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    cap.release()
    return num_frames

def set_io_level(io):
    if io.lower() == "quiet":
        Logger.set_io_level(Logger.QUIET)
    elif io.lower() == "basic":
        Logger.set_io_level(Logger.BASIC)
    elif io.lower() == "warning":
        Logger.set_io_level(Logger.WARNING)
    elif io.lower() == "detail":
        Logger.set_io_level(Logger.DETAIL)
    elif io.lower() == "debug":
        Logger.set_io_level(Logger.DEBUG)
    else:
        Logger.basic("Invalid IO level: " + io + " Use warning.")
        Logger.set_io_level(Logger.WARNING)
