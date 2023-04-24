import PIL
import cv2 as cv
import os
import re
from typing import List, Tuple
import sys
import click


IMAGE_FORMAT=["jpg", "png", "tif", "tiff", "jpeg"]

@click.command()
@click.argument("input_dir")
@click.argument("output")
@click.option("--ext", type=str, default=None, help="Format of input images. If not specified, " \
              "use the first found valid image format.")
@click.option("--start-id", default=0, help="ID of the first image to be included in the video")
@click.option("--end-id", default=sys.maxsize, help="ID of the last image to be included in the video")
@click.option("--fps", default=24, help="Frames per second.")
@click.option("--res", nargs=2, type=int, default=None, help="Resolution of input image. If not "\
              "specified, use the dimensions of the first image.") # A tuple
@click.option("--format", default="avi", help="Default format of video")
def generate_video(input_dir, output, ext, start_id, end_id, fps, res, format):
    """
    Generate the video OUTPUT from all iamges in the INPUT_DIR folder.

    Only include [start_id, end_id] frames from the input folder.
    """
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
    files.sort(key=lambda f: int(re.match(".*_([a-zA-Z]*)([0-9]+)\.([a-z]+)", f).group(2)))
    files = [f for f in files if start_id <= int(re.match(".*_([a-zA-Z]*)([0-9]+)\.([a-z]+)", f).group(2)) <= end_id]

    if len(files) == 0:
        print(f"No valid image files found in {input_dir} with extension {ext}")

    images = [cv.imread(f) for f in files] # color pics are already in BGR order, not RBG
    if res == None:
        # In the format of (width, height) (or (column, row))
        # But the shape is (height, width)
        res = (images[0].shape[1], images[0].shape[0])

    # Only give a file name without extension.
    if "." not in output:
        output = "{:s}.{:s}".format(output, format)

    if format == "avi":
        fourcc = cv.VideoWriter_fourcc(*'XVID') # XVID: .avi; mp4v: .mp4
    elif format == "mp4":
        fourcc = cv.VideoWriter_fourcc(*'mp4v') # XVID: .avi; mp4v: .mp4
    else:
        fourcc = cv.VideoWriter_fourcc(*'XVID') # XVID: .avi; mp4v: .mp4
    
    print(fps)
    print(res)
    print(output)

    video = cv.VideoWriter(output, fourcc, fps, res) # res is (width, height)
    for i in images:
        video.write(i)
    video.release() # generate video.
    cv.destroyAllWindows()

if __name__ == "__main__":
    generate_video()