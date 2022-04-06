import unittest
from src.datagen.bead_gen import Beads, bead_data_to_file, numpy_to_maskrcnn_target, read_target_from_file
import os
import scipy.io as sio


class TestBeads(unittest.TestCase):
    def setUp(self):
        self.bead = Beads()

    def test_gen_sample(self):
        img, seg, bbox = self.bead.gen_sample()
        self.assertIsNotNone(img)
        self.assertIsNotNone(seg)
        self.assertIsNotNone(bbox)

    def test_bead_data_to_file(self):
        filename = os.path.join(os.getcwd(), "train")
        try:
            os.mkdir(filename)
        except:
            print("Folder already exists")

        bead_data_to_file(filename, N=100)

    def test_numpy_to_maskrcnn_target(self):
        filename = os.path.join(os.getcwd(), "train")
        seg, bbox = read_target_from_file(filename, 0)
        area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        target = numpy_to_maskrcnn_target(bbox, None, seg, 0, area)
