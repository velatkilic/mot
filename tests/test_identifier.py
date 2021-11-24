from unittest import TestCase
from src.mot.identifier import identify
import os


class Test(TestCase):
    def test_identify(self):
        cwd = os.getcwd()
        fname = os.path.join(cwd, "data/test.mp4")
        model = os.path.join(cwd, "data/")
        imgOutDir = os.path.join(cwd, "data/imgout")
        os.makedirs(imgOutDir, exist_ok=True)

        blobsOutFile = os.path.join(cwd, "data/blobsOutFile.dat")

        identify(fname, model, imgOutDir, blobsOutFile, crop=(0, 0, 512, 512))
