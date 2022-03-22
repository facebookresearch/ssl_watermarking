# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import root_scalar
from scipy.special import betainc
from scipy.stats import ortho_group

from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_backbone(path, name):
    """ Build a pretrained torchvision backbone from its name.

    Args:
        path: path to the checkpoint, can be an URL
        name: name of the architecture from torchvision (see https://pytorch.org/vision/stable/models.html) 
        or timm (see https://rwightman.github.io/pytorch-image-models/models/). 
        We highly recommand to use Resnet50 architecture as available in torchvision. 
        Using other architectures (such as non-convolutional ones) might need changes in the implementation.
    """
    if hasattr(models, name):
        model = getattr(models, name)(pretrained=True)
    else:
        import timm
        if name in timm.list_models():
            model = timm.models.create_model(name, num_classes=0)
        else:
            raise NotImplementedError('Model %s does not exist in torchvision'%name)
    model.head = nn.Identity()
    model.fc = nn.Identity()
    if path is not None:
        if path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(path, progress=False, map_location=device)
        else:
            checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint
        for ckpt_key in ['state_dict', 'model_state_dict', 'teacher']:
            if ckpt_key in checkpoint:
                state_dict = checkpoint[ckpt_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
    return model.to(device, non_blocking=True)

def get_linear_layer(weight, bias):
    """ Create a linear layer from weight and bias matrices """
    dim_out, dim_in = weight.shape
    layer = nn.Linear(dim_in, dim_out)
    layer.weight = nn.Parameter(weight)
    layer.bias = nn.Parameter(bias)
    return layer

def load_normalization_layer(path, mode='whitening'):
    """ Loads the normalization layer from a checkpoint and returns the layer. """
    checkpoint = torch.load(path, map_location=device)
    if mode=='whitening':
        # if PCA whitening is used scale the feature by the dimension of the latent space
        D = checkpoint['weight'].shape[1] 
        weight = torch.nn.Parameter(D*checkpoint['weight'])
        bias = torch.nn.Parameter(D*checkpoint['bias'])
    else:
        weight = checkpoint['weight']
        bias = checkpoint['bias']
    return get_linear_layer(weight, bias).to(device, non_blocking=True)

class NormLayerWrapper(nn.Module):
    """
    Wraps backbone model and normalization layer
    """
    def __init__(self, backbone, head):
        super(NormLayerWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        output = self.backbone(x)
        return self.head(output)

def cosine_pvalue(c, d, k=1):
    """
    Returns the probability that the absolute value of the projection between random unit vectors is higher than c
    Args:
        c: cosine value
        d: dimension of the features
        k: number of dimensions of the projection
    """
    assert k>0
    a = (d - k) / 2.0
    b = k / 2.0
    if c < 0:
        return 1.0
    return betainc(a, b, 1 - c ** 2)

def pvalue_angle(dim, k=1, angle=None, proba=None):
    """
    Links the pvalue to the angle of the hyperspace. 
    If angle is input, the function returns the pvalue for the given angle.
    If proba is input, the function returns the angle for the given FPR.
    Args:
        dim: dimension of the latent space
        k: number of axes of the projection
        angle: angle of the hyperspace
        proba: target probability of false positive 
    """
    def f(a):
        return cosine_pvalue(np.cos(a), dim, k) - proba
    a = root_scalar(f, x0=0.49*np.pi, bracket=[0, np.pi/2])
    return a.root

def generate_carriers(k, d, output_fpath=None):
    """
    Generate k random orthonormal vectors of size d. 
    Args:
        k: number of bits to watermark
        d: dimension of the watermarking space
        output_fpath: path where the tensor is saved
    Returns: 
        tensor KxD
    """
    assert k<=d
    if k==1:
        carriers = torch.randn(1, d)
        carriers /= torch.norm(carriers, dim=1, keepdim=True)
    else:
        carriers = ortho_group.rvs(d)[:k,:]
        carriers = torch.tensor(carriers, dtype=torch.float)
    if output_fpath is not None:
        torch.save(carriers, output_fpath)
    return carriers

def generate_messages(n, k):
    """
    Generate random original messages.
    Args:
        n: Number of messages to generate
        k: length of the message
    Returns:
        msgs: boolean tensor of size nxk
    """
    return torch.rand((n,k))>0.5

def string_to_binary(st):
    """ String to binary """
    return ''.join(format(ord(i), '08b') for i in st)

def binary_to_string(bi):
    """ Binary to string """
    return ''.join(chr(int(byte,2)) for byte in [bi[ii:ii+8] for ii in range(0,len(bi),8)] )

def get_num_bits(path, msg_type):
    """ Get the number of bits of the watermark from the text file """ 
    with open(path, 'r') as f:
        lines = [line.strip() for line in f]
    if msg_type == 'bit':
        return max([len(line) for line in lines])
    else:
        return 8*max([len(line) for line in lines])

def load_messages(path, msg_type, N):
    """ Load messages from a file """
    with open(path, 'r') as f:
        lines = [line.strip() for line in f]
    if msg_type == 'bit':
        num_bit = max([len(line) for line in lines])
        lines = [line + '0'*(num_bit-len(line)) for line in lines]
        msgs = [[int(i)==1 for i in line] for line in lines]
    else:
        num_byte = max([len(line) for line in lines])
        lines = [line + ' '*(num_byte-len(line)) for line in lines]
        msgs = [[int(i)==1 for i in string_to_binary(line)] for line in lines]
    msgs = msgs * (N//len(msgs)+1)
    return torch.tensor(msgs[:N])

def save_messages(msgs, path):
    """ Save messages to file """
    txt_msgs = [''.join(map(str, x.type(torch.int).tolist())) for x in msgs]
    txt_msgs = '\n'.join(txt_msgs)
    with open(os.path.join(path), 'w') as f:
        f.write(txt_msgs)

def parse_params(s):
    """
    Parse parameters into a dictionary, used for optimizer and scheduler parsing.
    Example: 
        "SGD,lr=0.01" -> {"name": "SGD", "lr": 0.01}
    """
    s = s.replace(' ', '').split(',')
    params = {}
    params['name'] = s[0]
    for x in s[1:]:
        x = x.split('=')
        params[x[0]]=float(x[1])
    return params


def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')
