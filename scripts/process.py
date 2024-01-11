import cv2 as cv
import os
import re
import sys
import click
import natsort
from pathlib import Path
from xmot.utils.image_utils import subtract_brightfield_by_shifting as sbf_shifting
from xmot.utils.image_utils import subtract_brightfield_by_scaling as sbf_scaling
from xmot.config import IMAGE_FORMAT

# Global variables. Simpler alternatives than passing click.context object.
orig_images = [] # A list of OpenCV image object (i.e. numpy.ndarray).
                 # It's a global variable to offer easy access of input images loaded
                 # in the parent command.
orig_image_names = [] # File name of the input images, including the ids.
orig_ext = None

@click.group()
@click.argument("input_dir")
@click.option("--start-id", default=0, help="ID of the first image to be included in the video")
@click.option("--end-id", default=sys.maxsize, help="ID of the last image (inclusive) to be included in the video")
@click.option("--ext", type=str, default=None, help="Format of input images. If not specified, " \
              "use the first found valid image format.")
@click.option('--crop', type=int, multiple=True, default=None, help='4 integers specifying the crop size in x1, y1, x2, y2 order.')
def process(input_dir, start_id, end_id, ext, crop):
    """
    Common batch processing routines of video images in INPUT_DIR.

    """
    # This function performs as a common entry point of all subcommands. It load all images
    # in the INPUT_DIR and assign it to the global "orig_images" variable.
    global orig_images, orig_ext, orig_image_names
    if ext == None:
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
        files = [f for f in files if os.path.isfile(f)]
        for f in files:
            if f.split(".")[-1] in IMAGE_FORMAT:
                ext = f.split(".")[-1]
                break
        files = [f for f in files if f.endswith(ext)]
    else:
        files = [f for f in os.listdir(input_dir) if f.endswith(ext)]
    
    files = natsort.natsorted(files)
    #files.sort(key=lambda f: int(re.match(".*_([a-zA-Z]*)([0-9]+)\.([a-z]+)", f).group(2)))
    
    files = [f for f in files if start_id <= int(re.match(".*_([a-zA-Z]*)([0-9]+)\.([a-zA-Z]+)", f).group(2)) <= end_id]

    if len(files) == 0:
        print(f"No valid image files found in {input_dir} with extension {ext}")

    orig_images = [cv.imread(f) for f in files] # color pics are already in BGR order, not RBG
    if crop != None and len(crop) == 4:
        print("Cropping images with rectangule: " + str(crop))
        x1, y1, x2, y2 = crop
        orig_images = [img[y1:y2, x1:x2, :] for img in orig_images]
    orig_image_names = [Path(f).resolve().name for f in files]
    orig_ext = ext

@process.command()
@click.argument("output_dir")
@click.option("--prefix", default="img", help="Prefix of the names of output images.")
@click.option("--new-start-id", default=0, help="ID of the first output image.")
@click.option("--new-ext", default=None, help="Format of output images. "\
              "Default is to keep the format of input image.")
def rename_images(output_dir, prefix, new_start_id, new_ext):
    global orig_images, orig_ext
    id = new_start_id
    ext = new_ext if new_ext != None else orig_ext

    for img in orig_images:
        cv.imwrite(f"{output_dir}/{prefix}_{id}.{ext}", img)
        id = id + 1

@process.command()
@click.argument("output_dir")
@click.argument("new_ext")
@click.option("--prefix", default=None, help="New prefix of the names of output images.")
def format_transform(output_dir, prefix, new_ext):
    """
    Tranform the input images from INPUT_DIR to new format NEW_EXT.
    
    Image names are kept intact. If PREFIX is specified, prefix of old names will be replaced, 
    but id will be kept.
    """
    global orig_images, orig_image_names
    for img, name in zip(orig_images, orig_image_names):
        if prefix != None:
            obj = re.match(".*_([a-zA-Z]*)([0-9]+)\.([a-zA-Z]+)", name).group(2)
            name = f"{prefix}_{obj.group(1)}{obj.group(2)}"
        else:
            name = name.split(".")[0] # <name>.<old_ext>
        cv.imwrite(f"{output_dir}/{name}.{new_ext}", img)


@process.command()
@click.argument("output_dir")
@click.argument("brightfield")
@click.option("--scale", default=1.0, type=float, help="Ratio of the peak pixel distribution of "\
              "brightfieldand image relative to video image.")
@click.option("--use-scale", is_flag=True, help="Use scaling to subtract brightfield.")
@click.option("--shift-back", is_flag=True, help="Shift the peak of pixel distribution back to that "\
              "of the orignl image. Default is False, since GMM works better on non-shifted images.")
@click.option("--prefix", default=None, help="Prefix of the names of output images. If used, assume"\
              " input image names follow the pattern <old_prefix>_<id>.<ext>.")
@click.option('--crop', type=int, multiple=True, default=None, help='4 integers specifying the crop size in x1, y1, x2, y2 order.')
def subtract_brightfield(output_dir, brightfield, prefix, scale, use_scale, shift_back, crop):
    """
    Subtracting BRIGHTFILED image from input images and write out to OUTPUT_DIR.

    Only works for grayscale image for now.
    """
    if use_scale:
        print("Note: Use the scaling approach to subtract brightfield image.")
    global orig_images, orig_ext, orig_image_names
    img_bf = cv.imread(brightfield, cv.IMREAD_GRAYSCALE)
    if crop != None and len(crop) == 4:
        print("Cropping brightfield image with : " + str(crop))
        x1, y1, x2, y2 = crop
        img_bf = img_bf[y1:y2, x1:x2]
    
    if prefix == None:
        image_names = orig_image_names
    else:
        image_names = [f"{prefix}_{name.split('_')[-1]}" for name in orig_image_names]

    for i in range(len(orig_images)):
        img = cv.cvtColor(orig_images[i], cv.COLOR_BGR2GRAY)
        if not use_scale:
            # This is the default option. Simply shift the brightfield image rather than changing
            # the shape of the pixel histogram
            img_subtracted, *_ = sbf_shifting(img, img_bf, scale=scale, shift_back=shift_back)
        else:
            img_subtracted, *_ = sbf_scaling(img, img_bf, scale=scale, shift_back=shift_back)
        cv.imwrite(os.path.join(output_dir, orig_image_names[i]), img_subtracted)

@process.command()
@click.argument("output_dir")
def invert(output_dir):
    """
    Batch invert pictures. Assume all pictures are in grayscale.
    """
    global orig_images, orig_ext, orig_image_names
    for i in range(len(orig_images)):
        img_inverse = cv.bitwise_not(orig_images[i])
        cv.imwrite(os.path.join(output_dir, orig_image_names[i]), img_inverse)

if __name__ == "__main__":
    process()