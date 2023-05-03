"""
Style transfer on synthetic spherical particles.
"""

import os
import numpy as np
import pathlib
from PIL import ImageOps

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from xmot.datagen.style_transfer import run_style_transfer
from xmot.datagen.bead_gen import *

from xmot.logger import Logger

class StyleDatasetGen:
    def __init__(self, bead_gen, style_imgs, model=None, outFolder=None, N=1000, gray=True, save_orig=False):
        """
        Arguments:
            bead_gen:   Beads  Generator of spherical particles.
            style_imgs: xmot.dataset.Dataset   Dataset of style images. It's a generic loader of
                                               images from video or folder of images.
        
        Options:
            save_orig:  bool   Save the content image before style transfer
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = N
        self.save_orig=save_orig

        # Set dataset
        if style_imgs is None:
            Logger.error("Style transfer dataset generator requires a dataset for style images.")
        else:
            self.style_imgs = style_imgs

        # Set output folder for the generated data
        if outFolder is None:
            self.outFolder = os.path.join(os.getcwd(), "train_style")
        else:
            self.outFolder = outFolder

        # Create folder if is doesn't exist
        try:
            os.mkdir(self.outFolder)
        except:
            Logger.warning("Folder of style-transfered beads already exists! Overwriting existing data.")

        # Bead generator
        self.bead_gen = bead_gen

        # use only features which has the CNNs (as opposed to the det heads)
        if model is not None:
            # Add this option primarily for HPC environment, which network might not be available.
            vgg19 = models.vgg19()
            state_dict = torch.load(model)
            vgg19.load_state_dict(state_dict)
            self.cnn = vgg19.features.to(self.device).eval()
        else:
            # If model path not provided, try to download from PyTorch website or load from cache.
            self.cnn = models.vgg19(weights='IMAGENET1K_V1').features.to(self.device).eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.tsf_expr = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(self.bead_gen.side),
            transforms.ToTensor()
        ])

        self.tsf_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        self.unloader = transforms.ToPILImage()  # reconvert into PIL image
        self.gray = gray

    def tensor2image(self, tensor):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unloader(image)

        if self.gray:
            image = ImageOps.grayscale(image)

        return image

    def gen_dataset(self):
        # train set
        dataset_dicts = []

        for i in range(self.N):
            content_img, seg, bbox = self.bead_gen.gen_sample()
            if self.save_orig:
                cv.imwrite(os.path.join(self.outFolder, f"syn_bead_{i}_orig.png"), content_img)
            ind = np.random.randint(0, self.style_imgs.length())

            style_img = self.tsf_expr(self.style_imgs.get_img(ind)).unsqueeze(0).to(self.device, torch.float)
            content_img = self.tsf_mask(content_img).unsqueeze(0).to(self.device, torch.float)

            input_img = content_img.clone()

            output = run_style_transfer(self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
                                        content_img, style_img, input_img)

            fname = os.path.join(self.outFolder, f"syn_bead_{i}.png")
            img_sty = self.tensor2image(output)
            img_sty.save(fname)
            write_target_to_file(seg, bbox, self.outFolder, i)