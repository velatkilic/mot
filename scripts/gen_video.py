import PIL
import cv2 as cv
from os import listdir, path
from typing import List, Tuple
import sys

"""
    Concatenate a set of images to generate a video.
    
    It can be used as a function or a script. When used as a script, the command is:
        python3 gen_video.py <image dir> <prefix> <ext> <output_path> 
                             <starting_number_of_filename> <ending_number>
                             <fps> <width> <height>
    
    Args:
        <image dir>: Path of folder containing all the images.
        <prefix>: Prefix of the image file names. E.g. the prefix for "frame_1.png",
            "frame_2.png" is "frame_"
        <ext>: Entension of iamge file names. E.g. the ext for "frame_1.png" is "png".
        <output_path>: Path to store the video output.
        <starting_number_of_filename>: Which image to start for the video. E.g. if
            giving 9, "frame_9.png" will be the first image to start.
        <ending_number>: Same as <starting_number_of_filename>. (inclusive)

    The <starting_number_of_filename>, <ending_number>, <fps>, <width> 
    and <height> are optional.

    Example:
        python3 gen_video.py ../tests/playback_test300AlZr/ plot_frame jpg  
            ../tests/playback_test300AlZr/test.avi 9
"""

def generate_video(images: List[str], output: str, fps: int = 24, 
                   res: Tuple[int] = None, format: str = "mp4"):
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

# Run the file as a script.
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Need at least four arguments: input folder, prefix, ext, output video name.",
              file=sys.stderr)
        exit(1)
    else:
        prefix = sys.argv[2]
        ext = sys.argv[3]
        output = sys.argv[4]

    start_number = -1
    end_number = sys.maxsize
    fps = 24
    res = None
    if len(sys.argv) >= 6:
        start_number = int(sys.argv[5])
    if len(sys.argv) >= 7:
        end_number = int(sys.argv[6])
    if len(sys.argv) >= 8:
        fps=int(sys.argv[7])
    if len(sys.argv) >= 9:
        res = (int(sys.argv[8]), int(sys.argv[9]))

    files = [f for f in listdir(sys.argv[1]) 
             if f.startswith(prefix) and f.endswith(ext)]
    files.sort(key=lambda f: int(f.replace(prefix, "").replace("." + ext, "")))

    # Remove images with numbers less than starting number, if given.
    numbers = [int(f.replace(prefix, "").replace("." + ext, "")) for f in files]
    
    for i in range(numbers[0], start_number):
        files.remove("{:s}{:d}.{:s}".format(prefix, i, ext))
    for i in range(end_number + 1, numbers[-1] + 1): # end_number is inclusive
        files.remove("{:s}{:d}.{:s}".format(prefix, i, ext))
    
    images = [path.join(sys.argv[1], f) for f in files]
    generate_video(images, output, fps, res)