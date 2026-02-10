import imageio
#from skimage.io import imread_collection
import cv2 as cv
import numpy as np
from xmot.logger import Logger
from xmot.utils.image_utils import load_images_from_dir


class Dataset:
    def __init__(self, video_name=None, crop=None, image_folder=None):
        """
        Attributes:
            crop:    List[int]   Crop area specified in bbox format. (crop[0], crop[1]) is the
                                 top left corner of the crop. (crop[2], crop[3]) is the lower
                                 right corner of the crop.
        """
        if video_name is not None:
            self.reader = imageio.get_reader(video_name)
        elif image_folder is not None:
            # Somehow the imread_collection raise exception saying it can't load images from folder.
            #self.reader = imread_collection(image_folder, conserve_memory=False)
            # Read all images at once. May be slow. TODO: Optimize as an iterator.
            self.reader, img_names = load_images_from_dir(image_folder)
        else:
            Logger.error("Video name or image folder cannot both be empty")
        self.image_folder = image_folder
        self.video_name = video_name
        self.crop = crop

        self.gray = False

        # Test if gray
        tst = self.get_img(0)
        if len(tst.shape) != 3:
            self.gray = True

    def get_img(self, idx):
        if self.video_name is not None:
            img = self.reader.get_data(idx)
        else:
            img = self.reader[idx]

        # crop[0], crop[2] specifies the x coordinates, which are columns of the np.ndarray
        # crop[1], crop[3] specifies the y coordinates, which are rows of the np.ndarray
        if self.crop is not None:
            img = img[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2], ...]

        # if self.gray:
        #     img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        # Map to 8 bit integer (in case input is 16 bit tiff file)
        if img.dtype == 'uint16':
            img = img / 2**8
            img = img.astype(np.uint8)
        return img

    def length(self):
        if self.video_name is not None:
            return self.reader.count_frames()
        elif self.image_folder is not None:
            return len(self.reader)
        else:
            return 0

    # For compatibility with GMM constructor.
    def get_video_name(self):
        return self.video_name
    
    def get_crop(self):
        return self.crop
