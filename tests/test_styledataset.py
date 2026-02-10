import unittest
from xmot.datagen.style_data_gen import StyleDatasetGen
from xmot.dataset import Dataset
import os


class TestStyleDataset(unittest.TestCase):

    def test_gen_dataset(self):
        # create a dataset
        fnam = os.path.join(os.getcwd(), "data/Effect of Mg on AlZr/(Al8Mg)Zr_Full_20kfps_90kfps_20170309_DG_150mm_167_S1/*.tif")
        dset = Dataset(image_folder=fnam)

        # StyleDataset Generator
        sdset = StyleDatasetGen(dset=dset, len=10)
        sdset.gen_dataset()

        print("End.")