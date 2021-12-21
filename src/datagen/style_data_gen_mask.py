"""
Neural Transfer Using PyTorch
=============================


**Author**: `Alexis Jacq <https://alexis-jacq.github.io>`_

**Edited by**: `Winston Herring <https://github.com/winston6>`_

*** Edited by Velat Kilic on Dec 7, 2020
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

import imageio
import os
import numpy as np
from scipy.ndimage import gaussian_filter

import pycocotools

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


######################################################################
# Loss Functions


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


######################################################################
# Importing the Model

# use only features which has the CNNs (as opposed to the det heads)
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Additionally, VGG networks are trained on images with each channel
# normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
# We will use them to normalize the image before sending it into the network.
#

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# We need to add our content loss and style loss layers immediately after
# the convolution layer they are detecting. To do this we must create a new `
# `Sequential`` module that has content loss and style loss modules correctly inserted.

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


######################################################################
# Optimization

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e5, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


######################################################################
# Finally, we can run the algorithm.
#

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


from detectron2.structures import BoxMode

# train set
dataset_dicts = []

# create folder to save training data
try:
    os.mkdir(cwd + "/train/")
except:
    print("Warning: Train folder already exists")

import time

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