import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torchvision import transforms
import scipy.io as sio
import os
import cv2 as cv
import glob


class Beads:
    def __init__(self, side=256, beadradMax=10, beadradMin=3, numbeadsMax=20, numbeadsMin=10,
                 sigma=1, no_overlap=True):
        self.side = side # resolution. I.e. x, y dimension of the image.
        self.beadradMax = beadradMax
        self.beadradMin = beadradMin
        self.numbeadsMax = numbeadsMax
        self.numbeadsMin = numbeadsMin
        self.sigma = sigma
        self.no_overlap = no_overlap

    def gen_sample(self):
        # Exclude beadradMax, numbeadsMax
        numbeads = np.random.randint(self.numbeadsMin, self.numbeadsMax)
        beadrad = (self.beadradMax - self.beadradMin) * np.random.rand(numbeads, 1) + self.beadradMin

        # Make sure coordinates of grid points are integers. Also to keep consistent with the
        # "mask" and "cenx", "ceny" variables by excluding the end point.
        x, y = np.meshgrid(np.linspace(0, self.side, num = self.side, endpoint=False),
                           np.linspace(0, self.side, num = self.side, endpoint=False))

        mask = np.zeros((self.side, self.side), dtype=bool) # default is False.
        cenx = np.random.randint(0, self.side, size=(numbeads, 1))
        ceny = np.random.randint(0, self.side, size=(numbeads, 1))

        seg = []  # segmentation map

        if self.no_overlap:
            for i in range(numbeads):
                # Regenerate until no overlap with previous particles
                has_overlap = False
                for j in range(0, i):
                    if (cenx[i] - cenx[j]) ** 2 + (ceny[i] - ceny[j]) ** 2 <= (beadrad[i] + beadrad[j]) ** 2:
                        has_overlap = True
                        break
                while(has_overlap):
                    has_overlap = False
                    tempx = np.random.randint(0, self.side)
                    tempy = np.random.randint(0, self.side)
                    temprad = (self.beadradMax - self.beadradMin) * np.random.rand() + self.beadradMin
                    for j in range(0, i):
                        if (tempx - cenx[j]) ** 2 + (tempy - ceny[j]) ** 2 <= (temprad + beadrad[j]) ** 2:
                            has_overlap = True
                            break
                    if not has_overlap:
                        cenx[i] = tempx
                        ceny[i] = tempy
                        beadrad[i] = temprad

                dmmy = (x - cenx[i]) ** 2 + (y - ceny[i]) ** 2 <= (beadrad[i]) ** 2
                seg.append(dmmy)
                mask = np.logical_or(mask, dmmy) # Find the cumulative mask.
        else:
            # Allow overlapped beads.
            for i in range(numbeads):
                dmmy = (x - cenx[i]) ** 2 + (y - ceny[i]) ** 2 <= (beadrad[i]) ** 2
                seg.append(dmmy)
                mask = np.logical_or(mask, dmmy) # Find the cumulative mask.
        mask = np.logical_not(mask) # Reverse the bit. Background is white and is True. Particles are black and are False.
        img = 255 * mask.astype(np.uint8) # Make the mask a binary image. True -> 255, False -> 0.
        img = gaussian_filter(img, self.sigma) # Blur the particles.
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
    target['area'] = torch.tensor(np.array([area]), dtype=torch.float32)
    if iscrowd is None:
        target['iscrowd'] = torch.zeros((len(bbox),), dtype=torch.int64)
    else:
        target['iscrowd'] = torch.tensor(iscrowd, dtype=torch.int64)

    return target


def bead_data_to_file(filename, N=10000, side=256, beadradMax=10, beadradMin=3, numbeadsMax=20, numbeadsMin=10,
                      sigma=1, no_overlap=False):
    bead_gen = Beads(side, beadradMax, beadradMin, numbeadsMax, numbeadsMin, sigma, no_overlap=no_overlap)

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
    data = sio.loadmat(fname) # A dict with keys "seg" and "bbox". See write_target_to_file().
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
