# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from tqdm import tqdm
import os

import torch
from torchvision import transforms

import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_whitening_layer_params(features, dim_out):
    """
    Compute the weight and bias parameters of a linear layer to be used to whiten features.
    Args:
        features (tensor): Features to compute the whitening parameters on, NxD
        dim_out (int): whitening layer output feature size
    """
    mean = features.mean(dim=0, keepdim=True) # NxD -> 1xD
    features_centered = features - mean # NxD
    cov = features_centered.T @ features_centered
    # cov = torch.mm(features_centered.T, features_centered) # DxN @ NxD -> DxD
    e, v = torch.symeig(cov, eigenvectors=True)
    # e [D] and v [D, D] are in ascending order of e
    # select principal components: e[D_out], v[D, D_out]
    e = e[-dim_out:]
    v = v[:, -dim_out:] 
    L = torch.diag(1.0 / torch.sqrt(e))
    weight = torch.mm(L, v.T)
    bias = -torch.mm(mean, weight.T).squeeze(0)
    return weight, bias

def save_normalization_layer(norm_layer, filename):
    """
    Save the normalization layer to a filename
    """
    torch.save({"weight": norm_layer.weight, "bias": norm_layer.bias}, filename)

def create_normalization_layer_from_datadir(model, data_dir, transform, dim_out=None, batch_size=150):
    """
    Compute the normalization layer and add this layer at the end of the model.

    Args:
        model: Model that has 'fc' as last layer
        data_dir: Directory containing the images
        transform: Transformation to apply to images before feature extraction
        dim_out: Select the dim_out more important eigen vectors
        batch_size: Batch size to use for feature extraction
    """
    dataloader = utils_img.get_dataloader(data_dir, transform, batch_size=batch_size)
    fts = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            ft = model(images.to(device, non_blocking=True)) # BxCxWxH -> BxD
            fts.append(ft.cpu())
        fts = torch.cat(fts) # [BxD,....,BxD] -> NxD 

    if dim_out is None:
        dim_out = fts.shape[-1]
    weight, bias = compute_whitening_layer_params(fts, dim_out=dim_out)
    return utils.get_linear_layer(weight, bias)


if __name__ == '__main__':

    def get_parser():
        parser = argparse.ArgumentParser()
        # model params
        parser.add_argument("--model_name", type=str, default='resnet50', help="Marking network architecture. See https://pytorch.org/vision/stable/models.html and https://rwightman.github.io/pytorch-image-models/models/ (Default: resnet50)")
        parser.add_argument("--model_path", type=str, default="models/dino_r50_plus.pth", help="Path to the model (Default: /models/dino_r50_plus.pth)")
        # image tranform
        parser.add_argument("--img_size", type=int, default=256)
        parser.add_argument("--crop_size", type=int, default=224)
        # directories
        parser.add_argument("--large_data_dir", type=str, required=True)
        parser.add_argument("--output_dir", type=str, default='normlayers/')

        return parser

    params = get_parser().parse_args()
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    normlayer_path = os.path.join(params.output_dir, 'normlayer.pth')
    center_crop = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.crop_size),
        transforms.ToTensor(), 
        utils_img.NORMALIZE_IMAGENET,
    ])
    print('>>> Building backbone...')
    backbone = utils.build_backbone(path=params.model_path, name=params.model_name)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    print('>>> Building layer...')
    layer = create_normalization_layer_from_datadir(backbone, params.large_data_dir, transform=center_crop)
    save_normalization_layer(layer, normlayer_path)
    print('Saved normalization layer to {}'.format(normlayer_path))

