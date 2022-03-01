from src.dataset import Dataset
from src.mot.identifier import identify
import os

cwd = os.getcwd()
fnam = os.path.join(cwd,
                    "data/Effect of Mg on AlZr/(Al8Mg)Zr_Full_20kfps_90kfps_20170309_DG_150mm_167_S1/*.tif")
dset = Dataset(image_folder=fnam)

imgOutDir = os.path.join(cwd, "data/imgout")
os.makedirs(imgOutDir, exist_ok=True)

blobsOutFile = os.path.join(cwd, "data/blobsOutFile.dat")

identify(dset, imgOutDir, blobsOutFile, train_set=["train", "train_style"])