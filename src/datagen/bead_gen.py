import numpy as np
from scipy.ndimage import gaussian_filter


class Beads:
    def __init__(self, side=256, beadradMax=3, beadradMin=3, numbeadsMax=20, numbeadsMin=10, sigma=1):
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
