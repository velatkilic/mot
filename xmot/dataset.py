import imageio
from skimage.io import imread_collection
from xmot.logger import Logger
import cv2 as cv
import numpy as np


class Dataset:
    def __init__(self, video_name=None, crop=None, image_folder=None):
        if video_name is not None:
            self.reader = imageio.get_reader(video_name)
        elif image_folder is not None:
            self.reader = imread_collection(image_folder, conserve_memory=False)
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

        if self.crop is not None:
            img = img[self.crop[0]:self.crop[2], self.crop[1]:self.crop[3], ...]

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

