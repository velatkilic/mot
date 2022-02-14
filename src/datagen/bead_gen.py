import numpy as np
import pathlib
import os
from scipy.ndimage import gaussian_filter
import cv2 as cv
import pycocotools

from detectron2.structures import BoxMode

from src.logger import Logger

class Beads:
    def __init__(self, side=256, beadradMax=10, beadradMin=3, numbeadsMax=20, numbeadsMin=10, sigma=1):
        self.side = side
        self.beadradMax = beadradMax
        self.beadradMin = beadradMin
        self.numbeadsMax = numbeadsMax
        self.numbeadsMin = numbeadsMin
        self.sigma = sigma

    def gen_sample(self):
        numbeads = np.random.randint(self.numbeadsMin, self.numbeadsMax)
        beadrad = (self.beadradMax - self.beadradMin) * np.random.rand(numbeads, 1) + self.beadradMin

        x, y = np.meshgrid(np.linspace(0, self.side, self.side), np.linspace(0, self.side, self.side))

        mask = np.zeros((self.side, self.side), dtype=bool)
        cenx = np.random.randint(0, self.side, size=(numbeads, 1))
        ceny = np.random.randint(0, self.side, size=(numbeads, 1))

        seg = []  # segmentation map

        for i in range(numbeads):
            dmmy = (x - cenx[i]) ** 2 + (y - ceny[i]) ** 2 <= (beadrad[i]) ** 2
            seg.append(dmmy)
            mask = np.logical_or(mask, dmmy)
        mask = np.logical_not(mask)
        img = 255 * mask.astype(np.uint8)
        img = gaussian_filter(img, self.sigma)
        img = img.reshape((self.side, self.side, 1))
        img = img.repeat(3, axis=2)

        x1 = np.clip(cenx - beadrad, 0, self.side)
        x2 = np.clip(cenx + beadrad, 0, self.side)
        y1 = np.clip(ceny - beadrad, 0, self.side)
        y2 = np.clip(ceny + beadrad, 0, self.side)
        bbox = np.concatenate((x1, y1, x2, y2), axis=1)

        return img, seg, bbox

class BeadDataset:
    def __init__(self, outFolder=None, len=10000, gray=True):
        self.len = len
        self.beads = Beads()
        self.gray = gray

        # Set output folder for the generated data
        if outFolder is None:
            self.outFolder = os.path.join(pathlib.Path(os.getcwd()), pathlib.Path('train/'))
        else:
            self.outFolder = outFolder

        # Create folder if is doesn't exist
        try:
            os.mkdir(self.outFolder)
        except:
            Logger.warning("Folder already exists, overwriting existing data")

    def gen_dataset(self):
        # train set
        dataset_dicts = []

        for i in range(self.len):
            record = {}
            img, seg, bbox = self.beads.gen_sample()

            if self.gray:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            fname = self.outFolder + '/train_' + str(i) + '.png'
            cv.imwrite(fname, img)

            record["file_name"] = fname
            record["image_id"] = i
            record["height"] = self.beads.side
            record["width"] = self.beads.side

            objs = []
            for j in range(len(bbox)):
                obj = {
                    "bbox": bbox[j],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": pycocotools.mask.encode(np.asarray(seg[j], order="F")),
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs

            dataset_dicts.append(record)
        np.savez(self.outFolder + '/annot.npz', dataset_dicts=dataset_dicts)