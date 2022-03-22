# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_0bit(imgs, carrier, angle, model):
    """
    0-bit watermarking detection.

    Args:
        imgs: List of PIL images
        carrier: Hypercone direction 1xD
        angle: Angle of the hypercone
        model: Neural net model to extract the features

    Returns:
        List of decoded datum as a dictionary for each image.
        Contains the following fields:
            - R: acceptance function of the hypercone, positive when x lies in the cone, negative otherwise
            - log10_pvalue: log10 of the p-value, i.e. if we were drawing O(1/pvalue) random carriers, 
                on expectation, one of them would give an R bigger or equal to the one that is observed.
    """
    rho = 1 + np.tan(angle)**2
    decoded_data = []

    for ii, img in enumerate(imgs):
        img = utils_img.default_transform(img).unsqueeze(0).to(device, non_blocking=True) # 1xCxHxW
        ft = model(img) # 1xCxWxH -> 1xD
        dot_product = (ft @ carrier.T).squeeze() # 1xD @ Dx1 -> 1
        norm = torch.norm(ft, dim=-1) # 1xD -> 1 
        R = (rho * dot_product**2 - norm**2).item()
        cosine = torch.abs(dot_product/norm)
        log10_pvalue = np.log10(utils.cosine_pvalue(cosine.item(), ft.shape[-1]))
        decoded_data.append({'index': ii, 'R': R, 'log10_pvalue': log10_pvalue})
    
    return decoded_data


def decode_multibit(imgs, carrier, model):
    """
    multi-bit watermarking decoding.

    Args:
        imgs: List of PIL images
        carrier (tensor of size KxD): K carriers of dimension D, each one determines a bit
        model: Neural net model to extract the features

    Returns:
        List of decoded datum as a dictionary for each image.
        Contains the following fields:
            - msg: message extracted from the watermark as a tensor of booleans
    """
    decoded_data = []
    for ii, img in enumerate(imgs):
        img = utils_img.default_transform(img).unsqueeze(0).to(device, non_blocking=True) # 1xCxHxW
        ft = model(img) # 1xCxWxH -> 1xD
        dot_product = ft @ carrier.T # 1xD @ DxK -> 1xK
        msg = torch.sign(dot_product).squeeze() > 0
        msg = msg.cpu()
        decoded_data.append({'index': ii, 'msg': msg})

    return decoded_data
