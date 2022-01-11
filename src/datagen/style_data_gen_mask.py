"""
Style transfer on beads
"""

from __future__ import print_function

import torch
import torchvision.models as models

import torchvision.transforms as transforms
from style_transfer import run_style_transfer

import imageio
import os
import numpy as np
from scipy.ndimage import gaussian_filter

import pycocotools
from detectron2.structures import BoxMode
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mname', type=str, default='test', help="MP4 video filename")
parser.add_argument('-n', '--num', type=int, default=10000, help='number of frames for training')
args = parser.parse_args()

######################################################################
# Data

cwd = os.getcwd()  # Get current working directory
mname = cwd + '/' + args.mname + '.mp4'  # Movie name
vid = imageio.get_reader(mname, 'ffmpeg')  # Read video

leng = vid.count_frames()


def get_img(idx):
    img = vid.get_data(idx)
    return img[0:512, 0:512]


expr_imgs = np.zeros((leng, 512, 512, 3), dtype=np.uint8)
for i in range(leng):
    expr_imgs[i, ...] = get_img(i)

# bead image and label generation

side = 256
beadradMax = 10
beadradMin = 3
numbeadsMax = 20
numbeadsMin = 10
sigma = 1


def genSample():
    numbeads = np.random.randint(numbeadsMin, numbeadsMax)
    beadrad = (beadradMax - beadradMin) * np.random.rand(numbeads, 1) + beadradMin

    x, y = np.meshgrid(np.linspace(0, side, side), np.linspace(0, side, side))

    mask = np.zeros((side, side), dtype=bool)
    cenx = np.random.randint(0, side, size=(numbeads, 1))
    ceny = np.random.randint(0, side, size=(numbeads, 1))

    seg = []  # segmentation map for mask-RCNN

    for i in range(numbeads):
        dmmy = (x - cenx[i]) ** 2 + (y - ceny[i]) ** 2 <= (beadrad[i]) ** 2
        seg.append(dmmy)
        mask = np.logical_or(mask, dmmy)
    mask = np.logical_not(mask)
    img = 255 * mask.astype(np.uint8)
    img = gaussian_filter(img, sigma)
    img = img.reshape((side, side, 1))
    img = img.repeat(3, axis=2)

    x1 = np.clip(cenx - beadrad, 0, side)
    x2 = np.clip(cenx + beadrad, 0, side)
    y1 = np.clip(ceny - beadrad, 0, side)
    y2 = np.clip(ceny + beadrad, 0, side)
    bbox = np.concatenate((x1, y1, x2, y2), axis=1)

    return img, seg, bbox


img, seg, bbox = genSample()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################################################################
# Style transfer

# use only features which has the CNNs (as opposed to the det heads)
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

tsf_expr = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(side),
    transforms.ToTensor()])

tsf_mask = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()])

unloader = transforms.ToPILImage()  # reconvert into PIL image

def ten2im(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    return image

# train set
dataset_dicts = []

# create folder to save training data
try:
    os.mkdir(cwd + "/train/")
except:
    print("Warning: Train folder already exists")

t1 = time.time()
for i in range(args.num):
    record = {}
    content_img, seg, bbox = genSample()
    ind = np.random.randint(0, leng)
    style_img = tsf_expr(expr_imgs[ind, ...]).unsqueeze(0).to(device, torch.float)
    content_img = tsf_mask(content_img).unsqueeze(0).to(device, torch.float)

    input_img = content_img.clone()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
    fname = cwd + '/train/train' + str(i) + '.jpg'

    img_sty = ten2im(output)
    img_sty.save(fname)

    record["file_name"] = fname
    record["image_id"] = i
    record["height"] = side
    record["width"] = side

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
np.savez(cwd + '/train/annot', dataset_dicts=dataset_dicts)

t2 = time.time()

print('Elapsed time: ' + str(t2 - t1))