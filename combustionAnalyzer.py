#!/usr/bin/python3
import os
import click
import cv2 as cv
from PIL import Image
import numpy as np
from typing import List

from src.mot.identifier import identify
from src.digraph.digraph import Digraph
from src.digraph.utils import load_blobs_from_text, collect_images, paste_images, generate_video
from src.digraph import commons
from src.logger import Logger

@click.command()
@click.argument("video")
@click.argument("model")
@click.argument("param_dir")
@click.argument("output_dir")
@click.option("-m", "--write-meta-data", "write_meta", is_flag=True, default=True,
              help="Whether write meta-data like bbox of identified particles "\
                   "and reproduced pictures from the diagraph.")
@click.option("-c", "--crop", default=(0, 0, 512, 512), type=(int, int, int, int),
              help="Crop sizes by coordinates of top-left and bottom-right corners" 
                   "for each video frame.")
@click.option("-g", "--gpu/--no-gpu", is_flag=True, default=True, help="Wehter use GPU in pyTorch")
@click.option("-d", "--draw-type", default="plain", type=str,
              help="Modes of pictures to reproduce: 'plain', 'overlay', 'line'."\
                   "'plain' means regular picture with particles drawn as circles filling bbox."\
                   "'overlay' means particles of all frames are drawn on the same picture."\
                   "'line' means particles are drawn in lines to emphasize trajectories.")
@click.option("-o", "--io", default="warning", type=str,
              help="IO level. Available options are: 'quiet', 'basic' "\
                   "'warning', 'detail', 'debug'")
@click.option("-v", "--verbose", is_flag=True, default=False)
def combustionAnalyzer(video, model, param_dir, output_dir, write_meta, crop, gpu, draw_type, io, verbose):
    """
    Identify particles from videos of combustion via a specified MODEL using parameters in PARAM_DIR. 
    Then track and analyze trajectories of particles using Kalmen filter and digraph.

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
    # Initiate parameters:
    blobsFile = os.path.join(output_dir, "blobs.txt")
    reproduced_video = os.path.join(output_dir, "reproduced.avi")
    set_io_level(io)

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

    Logger.basic("Identifying particles in video: {:s}".format(video))
    num_frames = count_video_frame(video)
    # We pass on the video path since each function needs its own video pointer.
    identify(video, model, detection_img_dir, blobsFile, paramDir=param_dir, gpu=gpu, crop=crop)

    Logger.basic("Reading identified particles ...")
    particles = load_blobs_from_text(blobsFile)

    Logger.detail("Loading particles into digraph ...")
    dg = Digraph()
    commons.PIC_DIMENSION = [crop[1], crop[3]]
    dg.add_video(particles) # Load particles identified in the video.

    Logger.detail("Detailed information of digraph: \n" + str(dg))

    Logger.basic("Drawing reproduced images ...")
    draw_id = True if verbose else False
    if draw_type.lower() == "plain":
        rep_imgs = dg.draw(reproduce_img_dir, write_meta, draw_id)
    elif draw_type.lower() == "overlay":
        rep_imgs = dg.draw_overlay(reproduce_img_dir, write_meta, draw_id)
    elif draw_type.lower() == "line":
        rep_imgs = dg.draw_line_format(reproduce_img_dir, write_meta, draw_id)
    else:
        Logger.warning("Invalid drawing mode: " + draw_type + " Use plain.")
        rep_imgs = dg.draw(reproduce_img_dir, write_meta, draw_id)

    Logger.basic("Reproducing video ...")
    orig_imgs = collect_images(detection_img_dir, "{:s}_".format(model), "jpg", 0, num_frames - 1)
    merged_imgs = paste_images(orig_imgs, rep_imgs, merged_img_dir, write_meta)

    cv_imgs = pillow2cv(merged_imgs)
    generate_video(cv_imgs, reproduced_video)

def count_video_frame(video) -> int:
    cap = cv.VideoCapture(video)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
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

def pillow2cv(pil_imgs: List[Image.Image]):
    """
    Convert image object in PILLOW Image format to opencv format (numpy arrays).
    """
    cv_imgs = [cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR) for img in pil_imgs]
    return cv_imgs

if __name__ == "__main__":
    combustionAnalyzer()