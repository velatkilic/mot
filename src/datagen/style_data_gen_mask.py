"""
Style transfer on beads
"""

import os
import numpy as np
import pathlib
from PIL import ImageOps

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import pycocotools
from detectron2.structures import BoxMode

from src.datagen.style_transfer import run_style_transfer
from src.datagen.bead_gen import Beads

from src.logger import Logger


class StyleDataset:
    def __init__(self, dset=None, outFolder=None, len=1000, gray=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.len = len

        # Set dataset
        if dset is None:
            Logger.error("Style transfer dataset generator requires a dataset for style images.")
        else:
            self.dset = dset

        # Set output folder for the generated data
        if outFolder is None:
            self.outFolder = os.path.join(pathlib.Path(os.getcwd()), pathlib.Path('train_style/'))
        else:
            self.outFolder = outFolder

        # Create folder if is doesn't exist
        try:
            os.mkdir(self.outFolder)
        except:
            Logger.warning("Folder already exists, overwriting existing data")

        # Bead generator
        self.beads = Beads()

        # use only features which has the CNNs (as opposed to the det heads)
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.tsf_expr = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(self.beads.side),
            transforms.ToTensor()
        ])

        self.tsf_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        self.unloader = transforms.ToPILImage()  # reconvert into PIL image
        self.gray = gray

    def ten2im(self, tensor):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unloader(image)

        if self.gray:
            image = ImageOps.grayscale(image)

        return image

    def gen_dataset(self):
        # train set
        dataset_dicts = []

        for i in range(self.len):
            record = {}
            content_img, seg, bbox = self.beads.gen_sample()
            ind = np.random.randint(0, self.dset.length())

            style_img = self.tsf_expr(self.dset.get_img(ind)).unsqueeze(0).to(self.device, torch.float)
            content_img = self.tsf_mask(content_img).unsqueeze(0).to(self.device, torch.float)

            input_img = content_img.clone()

            output = run_style_transfer(self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
                                        content_img, style_img, input_img)

            fname = self.outFolder + '/train_' + str(i) + '.png'
            img_sty = self.ten2im(output)
            img_sty.save(fname)

            record["file_name"] = fname
            record["image_id"] = i
            record["height"] = self.beads.side
            record["width"] = self.beads.side

            objs = []
            for j in range(len(bbox)):
                obj = {
                    "bbox": bbox[j],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": pycocotools.mask.encode(np.asarray(seg[j], order="F")),
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs

            dataset_dicts.append(record)
        np.savez(self.outFolder + '/annot.npz', dataset_dicts=dataset_dicts)
