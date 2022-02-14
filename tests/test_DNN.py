from unittest import TestCase

import os
from src.dataset import Dataset
from src.mot.detectors import DNN


class TestDNN(TestCase):
    def test_DNN_with_tif_dataset(self):
        cwd = os.getcwd()
        fnam = os.path.join(cwd,
                            "data/Effect of Mg on AlZr/(Al8Mg)Zr_Full_20kfps_90kfps_20170309_DG_150mm_167_S1/*.tif")
        dset = Dataset(image_folder=fnam)

        dnn = DNN(dset=dset)
        bbox, mask = dnn.predict(0)
