from unittest import TestCase
import os

from xmot.dataset import Dataset
from xmot.mot.detectors import *


class TestDNN(TestCase):
    def test_predict_first_image(self):
        # dataset
        fnam = os.path.join(os.getcwd(), "data/test.mp4")
        dset = Dataset(video_name=fnam)

        # DNN
        fname = os.path.join(os.getcwd(), "data/")
        dnn = DNN(fname=fname, dset=dset)
        bbox, mask = dnn.predict(0)
        self.assertIsNotNone(bbox)
        self.assertIsNotNone(mask)
