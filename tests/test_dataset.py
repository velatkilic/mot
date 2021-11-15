import unittest
from src.dataset import Dataset
import os


class TestDataset(unittest.TestCase):

    def test_read_frame_mp4(self):
        fnam = os.path.join(os.getcwd(), "data/test.mp4")
        dset = Dataset(video_name=fnam)
        img = dset.get_img(0)
        self.assertIsNotNone(img)

    def test_read_frame_avi(self):
        fnam = os.path.join(os.getcwd(), "data/playback_test300AlZr.avi")
        dset = Dataset(video_name=fnam)
        img = dset.get_img(0)
        self.assertIsNotNone(img)

    def test_dataset_length(self):
        fnam = os.path.join(os.getcwd(), "data/playback_test300AlZr.avi")
        dset = Dataset(video_name=fnam)
        self.assertGreater(dset.length(), 0)
