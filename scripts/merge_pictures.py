from PIL import Image
import sys
from os import listdir, path
from typing import List

"""
    A script to paste reproduced image side by side with the original image.

    Usage:
        python3 merge_pictures.py <dest_dir> <orig_dir> <orig_prefix> <orig_ext>
            <reproduced_dir> <rep_prefix> <rep_ext> 
            <staring_number> <ending_number>

    Note:
    1. <starting_number> and <ending_number> are inclusive.
    2. prefix and ext are prefix name for image files and ext is the extension of 
       image files. E.g. for "plot_frame9.jpg", prefix is "plot_frame" and ext is
       "jpg".
    3. Resolution of images are taken from images. Supposedly, the original and
       reproduced pictures have the same resolution.
    4. Merged picture will be called <dest_dir>/merged_1.png and so on.
    5. Example:
        python3 ../bin/merge_pictures.py ./reproduced/merged_overlaid_line/ 
            ./playback_test300AlZr plot_frame jpg ./reproduced/overlaid_line/ frame_ png 9 301
"""

def help():
    print("python3 merge_pictures.py <dest_dir> <orig_dir> <orig_prefix> <orig_ext> " +
          "<reproduced_dir> <rep_prefix> <rep_ext> " +
          "[<staring_number>] [<ending_number>]")

def collect_images(dir: str, prefix: str, ext: str, start: int, end: int) \
    -> List[Image.Image]:
    files = [f for f in listdir(dir) 
              if f.startswith(prefix) and f.endswith(ext)]
    files.sort(key=lambda f: int(f.replace(prefix, "").replace("." + ext, "")))
    numbers = [int(f.replace(prefix, "").replace("." + ext, "")) for f in files]
    if start != -1:
        # There is no guarantee the file names start with number 1, so use remove()
        # instead of del, which requires an index.
        for i in range(numbers[0], start):
            files.remove("{:s}{:d}.{:s}".format(prefix, i, ext))

    if end != sys.maxsize:
        for i in range(end + 1, numbers[-1]): # end is inclusive
            files.remove("{:s}{:d}.{:s}".format(prefix, i, ext))

    images = [Image.open(path.join(dir, f)) for f in files]
    return images


if len(sys.argv) < 10:
    print("Needs at least 9 arguments.")
    help()

dest_dir = sys.argv[1]
orig_dir = sys.argv[2]
orig_prefix = sys.argv[3]
orig_ext = sys.argv[4]
rep_dir = sys.argv[5]
rep_prefix = sys.argv[6]
rep_ext = sys.argv[7]
start_number = int(sys.argv[8])
end_number = int(sys.argv[9])

orig_images = collect_images(orig_dir, orig_prefix, orig_ext, start_number, end_number)
rep_images = collect_images(rep_dir, rep_prefix, rep_ext, start_number, end_number)

# original image and reproduced image should have same height.
new_res = (orig_images[0].width + rep_images[0].width, orig_images[0].height)

for i in range(len(orig_images)):
    im = Image.new("RGBA", new_res)
    im.paste(orig_images[i], box=(0, 0)) # top left cornor of the box to paste the picture.
    im.paste(rep_images[i], box=(orig_images[i].width + 1, 0))
    im.save(path.join(dest_dir, "merged_{:d}.png".format(start_number + i)))