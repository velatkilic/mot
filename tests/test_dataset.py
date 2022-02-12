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

    def test_read_frame_tif(self):
        fnam = os.path.join(os.getcwd(), "data/Effect of Mg on AlZr/(Al8Mg)Zr_Full_20kfps_90kfps_20170309_DG_150mm_167_S1/*.tif")
        dset = Dataset(image_folder=fnam)
        img = dset.get_img(0)
        self.assertIsNotNone(img)

    def test_dataset_length(self):
        fnam = os.path.join(os.getcwd(), "data/playback_test300AlZr.avi")
        dset = Dataset(video_name=fnam)
        self.assertGreater(dset.length(), 0)
