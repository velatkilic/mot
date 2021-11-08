import unittest
from src.dataset import Dataset
import os


class TestDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fnam = os.path.join(os.getcwd(), "data/test.mp4")
        self.dset = Dataset(self.fnam)

    def test_read_frame(self):
        img = self.dset.get_img(0)
        self.assertIsNotNone(img)

    def test_dataset_length(self):
        self.assertGreater(self.dset.length(), 0)
