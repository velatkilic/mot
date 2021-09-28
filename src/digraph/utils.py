import sys
import math
from typing import List, Tuple
import pandas as pd
from os import path, listdir
from PIL import Image
import cv2 as cv

from logger import Logger
from digraph.particle import Particle

CLOSE_IN_TIME = 10
CLOSE_IN_SPACE = 40

def distance(a: List[float], b: List[float]) -> float:
    """L2 norm of vectors of any dimension."""
    if len(a) != len(b):
        Logger.error("Cannot calculate distance between two vectors of different dimensions: " + \
                     "{:d} {:d}".format(len(a), len(b)))
        return -1
    sum = 0
    for i in range(0, len(a)):
        sum += (a[i] - b[i])**2
    return math.sqrt(sum)

def ptcl_distance(p1, p2):
    return distance(p1.get_position(), p2.get_position())

def traj_distance(t1, t2) -> float:
    """
        Calculate the nearest distance between two trajectories during the video.

        If t1 ends before (starts late) than t2, calculate the distance between the 
        end (start) of t1 and start (end) of t2. If two trajectory are too far away
        in time (difference larger than self.CLOSE_IN_TIME), they shouldn't have
        any relation and distance is set to infinity.

        If t1 and t2 coexist for some time, calculate the nearest distance during these
        frames of the video.
    """
    t1.sort_particles()
    t2.sort_particles()
    if t1.get_end_time() < t2.get_start_time():
        if t2.get_start_time() - t1.get_end_time() <= CLOSE_IN_TIME:
            return distance(t1.get_end_position(), t2.get_start_position())
        else:
            return float("inf")
    elif t1.get_start_time() > t2.get_end_time():
        if t1.get_start_time() - t2.get_end_time() <= CLOSE_IN_TIME:
            return distance(t1.get_start_position(), t2.get_end_position())
        else:
            return float("inf")
    else:
        start_time = max(t1.get_start_time(), t2.get_start_time())
        end_time = min(t1.get_end_time(), t2.get_end_time())
        t1_ptcls = t1.get_snapshots(start_time, end_time)
        t2_ptcls = t2.get_snapshots(start_time, end_time)
        min_dist = float("inf")
        for p1 in t1_ptcls:
            last_index = -1
            # t1, t2 are sorted. So the particle of t2 that exists at same time as next particle
            # of t1 must have larger index.
            for i in range(last_index + 1, len(t2_ptcls)):
                p2 = t2_ptcls[i]
                if p2.time_frame == p1.time_frame:
                    dist = distance(p1.get_position(), p2.get_position())
                    min_dist = dist if dist < min_dist else min_dist
                    last_index = i
                    break
        return min_dist

def load_excel(file_name: str) -> List[Particle]:
    """ A temporary io function to load data from Kerri-Lee's excel data.

    The function assumes a specific format of the excel data, and will be replaced by more 
    general format later.
    """
    data_id = pd.read_excel(file_name, sheet_name="Particle ID", engine="openpyxl")
    data_pos = pd.read_excel(file_name, sheet_name="Raw_data", engine="openpyxl")

    # remove N/A
    data_id = data_id.fillna(0).astype(int)

    particles = []
    for i in range(0, len(data_id)):
        row_id = data_id.loc[i]
        row_pos = data_pos.loc[i]

        row_id = row_id[row_id.gt(0)]
        if len(row_id) <= 1:
            # Has only "Act_frame" column and no ided particle in this frame.
            continue

        time_frame = row_id["Act_frame"]
        # Iterate all particles ided in this frame
        for j in range(1, len(row_id)):   
            id = row_id[j]
            pos = [row_pos[2 * id], row_pos[2 * id + 1]]
            # Has no bubble info and predicted positions for now.
            particles.append(Particle(id, time_frame, pos))
    
    return particles

def load_text(file_name: str) -> List[Particle]:
    particles = []
    with open(file_name, "r") as f:
        for line in f:
            terms = line.replace(" ", "").split(",")
            terms = [int(term) for term in terms]
            # <TODO> Add length check to prevent array out of size error
            position = [terms[0], terms[1]]   # x1, y1
            bbox = [terms[4], terms[5]]       # width, height
            idx = terms[6]
            time_frame = terms[7]
            particles.append(Particle(idx, time_frame, position, bbox=bbox))
    return particles

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

def paste_images(left_imgs: List[Image.Image], right_imgs: List[Image.Image], dest, write_img) \
    -> List[Image.Image]:

    images = []
    # original image and reproduced image should have same height.
    if len(left_imgs) != len(right_imgs):
        Logger.error("There aren't same number of original and reproduced iamges. Please check!")
    
    new_res = (left_imgs[0].width + right_imgs[0].width, left_imgs[0].height)
    for i in range(len(left_imgs)):
        im = Image.new("RGBA", new_res)
        im.paste(left_imgs[i], box=(0, 0)) # top left cornor of the box to paste the picture.
        im.paste(right_imgs[i], box=(left_imgs[i].width + 1, 0))
        if write_img:
            im.save(path.join(dest, "merged_{:d}.png".format(i)))
        images.append(im)
    return images

def generate_video(images: List[str], output: str, fps: int = 24, 
                   res: Tuple[int] = None, format: str = "avi"):
    """
        Generate video from given list of iamges.
        
        Args:
            images: List of string representing paths of input iamges.
            output: Name of the video. If extension is given, format is ignored.
            fps: Frame rate of the video.
            res: Resolution of the video in pixel. If not given, the largest of
                all images are used to accommodate all images.
            format: Format of the video. Ignored when file extension is given in
                name.
    """
    ims = [cv.imread(i) for i in images] # color pics are already in BGR order, not RBG
    # Collect info from images
    # If resolution not given, use maximum size of all images.
    if res == None:
        # image are numpy.ndarray. image.shape = (height, width, number of color channels)
        max_height = max([i.shape[0] for i in ims])
        max_width = max([i.shape[1] for i in ims])
        res = (max_width, max_height)

    if "." not in output:
        output = "{:s}.{:s}".format(output, format)

    fourcc = cv.VideoWriter_fourcc(*'XVID') # XVID: .avi; mp4v: .mp4
    video = cv.VideoWriter(output, fourcc, fps, res) # res is (width, height)
    for i in ims:
        video.write(i)
    video.release() # generate video.
    cv.destroyAllWindows()