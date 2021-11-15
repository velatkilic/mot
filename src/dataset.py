import imageio
from skimage.io import imread_collection
from src.logger import Logger


class Dataset:
    def __init__(self, video_name=None, crop=None, image_folder=None):
        if video_name is not None:
            self.reader = imageio.get_reader(video_name)
        elif image_folder is not None:
            self.reader = imread_collection(image_folder, conserve_memory=False)
        else:
            Logger.error("Video name or image folder cannot both be empty")
        self.video_name = video_name
        self.crop = crop

    def get_img(self, idx):
        if self.video_name is not None:
            img = self.reader.get_data(idx)
        else:
            img = self.reader[idx]

        if self.crop is not None:
            img = img[self.crop[0]:self.crop[2], self.crop[1]:self.crop[3], ...]
        return img

    def length(self):
        if self.video_name is not None:
            return self.reader.count_frames()
        else:
            return len(self.reader)
