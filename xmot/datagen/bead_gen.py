import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torchvision import transforms
import scipy.io as sio
import os
import cv2 as cv
import glob


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


def numpy_to_maskrcnn_target(bbox, labels, seg, idx, area, iscrowd=None):
    target = {'boxes': torch.tensor(bbox, dtype=torch.float32)}
    if labels is None:
        target['labels'] = torch.ones(len(bbox), dtype=torch.int64)
    else:
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
    target['masks'] = torch.tensor(seg, dtype=torch.uint8)
    target['image_id'] = torch.tensor([idx])
    target['area'] = torch.tensor([area], dtype=torch.float32)
    if iscrowd is None:
        target['iscrowd'] = torch.zeros((len(bbox),), dtype=torch.int64)
    else:
        target['iscrowd'] = torch.tensor(iscrowd, dtype=torch.int64)

    return target


def bead_data_to_file(filename, N=10000, side=256, beadradMax=10, beadradMin=3, numbeadsMax=20, numbeadsMin=10,
                      sigma=1):
    bead_gen = Beads(side, beadradMax, beadradMin, numbeadsMax, numbeadsMin, sigma)

    for i in range(N):
        img, seg, bbox = bead_gen.gen_sample()
        iname = os.path.join(filename, "syn_bead_" + str(i) + ".png")
        cv.imwrite(iname, img)
        write_target_to_file(seg, bbox, filename, i)


def write_target_to_file(seg, bbox, filename, idx):
    fname = os.path.join(filename, "syn_bead_" + str(idx) + ".mat")
    sio.savemat(fname, {"seg": seg, "bbox": bbox})


def read_target_from_file(filename, idx):
    fname = os.path.join(filename, "syn_bead_" + str(idx) + ".mat")
    data = sio.loadmat(fname)
    return data["seg"], data["bbox"]


def collate_fn(batch):
    return tuple(zip(*batch))


class BeadDataset(torch.utils.data.Dataset):
    def __init__(self, bead_gen, length=10000, tsf=None):
        self.bead_gen = bead_gen
        self.length = length
        if tsf is None:
            self.tsf = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.tsf = tsf

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # generate data
        img, seg, bbox = self.bead_gen.gen_sample()

        # format target
        area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        target = numpy_to_maskrcnn_target(bbox, None, seg, idx, area)

        # transform image
        img = self.tsf(img)

        return img, target


class BeadDatasetFile(torch.utils.data.Dataset):
    def __init__(self, filename, tsf=None):
        self.filename = filename
        if tsf is None:
            self.tsf = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.tsf = tsf

        self.length = len(glob.glob(filename))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # read data
        seg, bbox = read_target_from_file(self.filename, idx)

        iname = os.path.join(self.filename, "syn_bead_" + str(idx) + ".png")
        img = cv.imread(iname)

        # format target
        area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        target = numpy_to_maskrcnn_target(bbox, None, seg, idx, area)

        # transform image
        img = self.tsf(img)

        return img, target
