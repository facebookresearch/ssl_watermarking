# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1).to(device)
image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1).to(device)
default_transform = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE_IMAGENET,
])

def normalize_img(x):
    """ Normalize image to approx. [-1,1] """
    return (x.to(device) - image_mean) / image_std

def unnormalize_img(x):
    """ Unnormalize image to [0,1] """
    return (x.to(device) * image_std) + image_mean

def round_pixel(x):
    """ 
    Round pixel values to nearest integer. 
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    x_pixel = 255 * unnormalize_img(x)
    y = torch.round(x_pixel).clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def project_linf(x, y, radius):
    """ 
    Clamp x so that Linf(x,y)<=radius
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        radius: Radius of Linf ball for the images in pixel space [0, 255]
     """
    delta = x - y
    delta = 255 * (delta * image_std)
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / image_std
    return y + delta

def psnr_clip(x, y, target_psnr):
    """ 
    Clip x so that PSNR(x,y)=target_psnr 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        target_psnr: Target PSNR value in dB
    """
    delta = x - y
    delta = 255 * (delta * image_std)
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    if psnr<target_psnr:
        delta = (torch.sqrt(10**((psnr-target_psnr)/10))) * delta 
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    delta = (delta / 255.0) / image_std
    return y + delta



class SSIMAttenuation:

    def __init__(self, window_size=17, sigma=1.5, device="cpu"):
        """ Self-similarity attenuation, to make sure that the augmentations occur high-detail zones. """
        self.window_size = window_size
        _1D_window = torch.Tensor(
            [np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]
            ).to(device, non_blocking=True)
        _1D_window = (_1D_window/_1D_window.sum()).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.window = Variable(_2D_window.expand(3, 1, window_size, window_size).contiguous())

    def heatmap(self, img1, img2):
        """
        Compute the SSIM heatmap between 2 images, based upon https://github.com/Po-Hsun-Su/pytorch-ssim 
        Args:
            img1: Image tensor with values approx. between [-1,1]
            img2: Image tensor with values approx. between [-1,1]
            window_size: Size of the window for the SSIM computation
        """
        window = self.window
        window_size = self.window_size
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = 3)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = 3)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = 3) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = 3) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = 3) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map

    def apply(self, x, y):
        """ 
        Attenuate x using SSIM heatmap to concentrate changes of y wrt. x around edges
        Args:
            x: Image tensor with values approx. between [-1,1]
            y: Image tensor with values approx. between [-1,1], ex: original image
        """
        delta = x - y
        ssim_map = self.heatmap(x, y) # 1xCxHxW
        ssim_map = torch.sum(ssim_map, dim=1, keepdim=True)
        ssim_map = torch.clamp_min(ssim_map,0)
        delta = delta*ssim_map
        return y + delta


def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.size][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.size][::-1]
    return functional.resize(x, new_edges_size)

def get_dataloader(data_dir, transform=default_transform, batch_size=128, shuffle=False, num_workers=4):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader

def pil_imgs_from_folder(folder):
    """ Get all images in the folder as PIL images """
    images = []
    filenames = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder,filename))
            if img is not None:
                filenames.append(filename)
                images.append(img)
        except:
            print("Error opening image: ", filename)
    return images, filenames