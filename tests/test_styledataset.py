import unittest
from src.datagen.style_data_gen_mask import StyleDataset
from src.dataset import Dataset
import os


class TestStyleDataset(unittest.TestCase):

    def test_gen_dataset(self):
        # create a dataset
        fnam = os.path.join(os.getcwd(), "data/Effect of Mg on AlZr/(Al8Mg)Zr_Full_20kfps_90kfps_20170309_DG_150mm_167_S1/*.tif")
        dset = Dataset(image_folder=fnam)

        # StyleDataset Generator
        sdset = StyleDataset(dset=dset, len=10)
        sdset.gen_dataset()

        print("lol")