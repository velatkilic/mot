from unittest import TestCase

from src.dataset import Dataset
from src.mot.identifier import identify
import os


class Test(TestCase):
    # def test_identify_mp4(self):
    #     cwd = os.getcwd()
    #
    #     crop = (0, 0, 512, 512)
    #     fname = os.path.join(cwd, "data/test.mp4")
    #     dset = Dataset(video_name=fname, crop=crop)
    #
    #     model = os.path.join(cwd, "data/")
    #     imgOutDir = os.path.join(cwd, "data/imgout")
    #     os.makedirs(imgOutDir, exist_ok=True)
    #
    #     blobsOutFile = os.path.join(cwd, "data/blobsOutFile.dat")
    #
    #     identify(dset, model, imgOutDir, blobsOutFile, crop=crop)
    #
    # def test_identify_tif(self):
    #     cwd = os.getcwd()
    #     fnam = os.path.join(cwd,
    #                         "data/Effect of Mg on AlZr/(Al8Mg)Zr_Full_20kfps_90kfps_20170309_DG_150mm_167_S1/*.tif")
    #     dset = Dataset(image_folder=fnam)
    #
    #     model = os.path.join(cwd, "data/")
    #     imgOutDir = os.path.join(cwd, "data/imgout")
    #     os.makedirs(imgOutDir, exist_ok=True)
    #
    #     blobsOutFile = os.path.join(cwd, "data/blobsOutFile.dat")
    #
    #     identify(dset, imgOutDir, blobsOutFile, model=model)

    def test_identify_avi_without_model_without_data(self):
        cwd = os.getcwd()
        # fnam = os.path.join(cwd,
        #                     "data/Effect of Mg on AlZr/(Al8Mg)Zr_Full_20kfps_90kfps_20170309_DG_150mm_167_S1/*.tif")
        # dset = Dataset(image_folder=fnam)

        crop = (0, 0, 512, 512)
        fname = os.path.join(cwd, "data/test.mp4")
        dset = Dataset(video_name=fname, crop=crop)

        imgOutDir = os.path.join(cwd, "data/imgout")
        os.makedirs(imgOutDir, exist_ok=True)

        blobsOutFile = os.path.join(cwd, "data/blobsOutFile.dat")

        identify(dset, imgOutDir, blobsOutFile)

    def test_identify_avi_without_model(self):
        cwd = os.getcwd()

        crop = (0, 0, 512, 512)
        fname = os.path.join(cwd, "data/test.mp4")
        dset = Dataset(video_name=fname, crop=crop)

        imgOutDir = os.path.join(cwd, "data/imgout")
        os.makedirs(imgOutDir, exist_ok=True)

        blobsOutFile = os.path.join(cwd, "data/blobsOutFile.dat")

        # train data
        train1 = os.path.join(cwd, "train")
        train2 = os.path.join(cwd, "train_style")

        identify(dset, imgOutDir, blobsOutFile, train_set=[train1, train2])
