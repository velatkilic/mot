import imageio
from src.logger import Logger


class Dataset:
    def __init__(self, filename=None, crop=None):
        if filename is None:
            Logger.error("Filename cannot be empty")
        self.crop = crop
        self.reader = imageio.get_reader(filename)

    def get_img(self, idx):
        img = self.reader.get_data(idx)
        if self.crop is not None:
            img = img[self.crop[0]:self.crop[2], self.crop[1]:self.crop[3], ...]
        return img

    def length(self):
        return self.reader.count_frames()
