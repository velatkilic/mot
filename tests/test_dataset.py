import unittest
from src.dataset import Dataset
import os


class MyTestCase(unittest.TestCase):
    def test_read_frame(self):
        fnam = os.path.join(os.getcwd(), "data/test.mp4")
        dset = Dataset(fnam)
        img  = dset.get_img(0)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
