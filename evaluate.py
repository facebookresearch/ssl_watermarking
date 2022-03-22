# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchvision.transforms import functional
from augly.image import functional as aug_functional

import decode

import utils_img
import utils

pd.options.display.float_format = "{:,.3f}".format
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attacks_dict = {
    "none": lambda x : x,
    "rotation": functional.rotate,
    "grayscale": functional.rgb_to_grayscale,
    "contrast": functional.adjust_contrast,
    "brightness": functional.adjust_brightness,
    "hue": functional.adjust_hue,
    "hflip": functional.hflip,
    "vflip": functional.vflip,
    "blur": functional.gaussian_blur, # sigma = ksize*0.15 + 0.35  - ksize = (sigma-0.35)/0.15
    "jpeg": aug_functional.encoding_quality,
    "resize": utils_img.resize,
    "center_crop": utils_img.center_crop,
    "meme_format": aug_functional.meme_format,
    "overlay_emoji": aug_functional.overlay_emoji,
    "overlay_onto_screenshot": aug_functional.overlay_onto_screenshot,
}

attacks = [{'attack': 'none'}] \
    + [{'attack': 'meme_format'}] \
    + [{'attack': 'overlay_onto_screenshot'}] \
    + [{'attack': 'rotation', 'angle': jj} for jj in range(0,45,5)] \
    + [{'attack': 'center_crop', 'scale': 0.1*jj} for jj in range(1,11)] \
    + [{'attack': 'resize', 'scale': 0.1*jj} for jj in range(1,11)] \
    + [{'attack': 'blur', 'kernel_size': 1+2*jj} for jj in range(1,10)] \
    + [{'attack': 'jpeg', 'quality': 10*jj} for jj in range(1,11)] \
    + [{'attack': 'contrast', 'contrast_factor': 0.5*jj} for jj in range(0,5)] \
    + [{'attack': 'brightness', 'brightness_factor': 0.5*jj} for jj in range(1,5)] \
    + [{'attack': 'hue', 'hue_factor': -0.5 + 0.25*jj} for jj in range(0,5)] \
    + [{'attack': 'hue', 'hue_factor': 0.2}] 
    
def generate_attacks(img, attacks):
    """ Generate a list of attacked images from a PIL image. """
    attacked_imgs = []
    for attack in attacks:
        attack = attack.copy()
        attack_name = attack.pop('attack')
        attacked_imgs.append(attacks_dict[attack_name](img, **attack))
    return attacked_imgs


def decode_0bit_from_folder(img_dir, carrier, angle, model):
    """
    Args:
        img_dir: Folder containing the images to decode
        carrier: Hypercone direction 1xD
        angle: Angle of the hypercone        
        model: Neural net model to extract the features

    Returns:
        df: Dataframe with the decoded message for each image
    """
    imgs, filenames = utils_img.pil_imgs_from_folder(img_dir)
    decoded_data = decode.decode_0bit(imgs, carrier, angle, model)
    df = pd.DataFrame(decoded_data)
    df['filename'] = filenames
    df['marked'] = df['R'] > 0
    df.drop(columns=['R', 'log10_pvalue'], inplace=True)
    return df


def evaluate_0bit_on_attacks(imgs, carrier, angle, model, params, attacks=attacks, save=True):
    """
    Args:
        imgs: Watermarked images, list of PIL Images
        carrier: Hypercone direction 1xD
        angle: Angle of the hypercone
        model: Neural net model to extract the features
        params: Must contain verbose, output_dir 
        attacks: List of attacks to apply
        save: Whether to save instances of attacked images for the first image

    Returns:
        df: Dataframe with the detection scores for each transformation
    """

    logs = []
    for ii, img in enumerate(tqdm(imgs)):
        
        attacked_imgs = generate_attacks(img, attacks)
        if ii==0 and save:
            imgs_dir = os.path.join(params.output_dir, 'imgs')
            for jj in range(len(attacks)):
                attacked_imgs[jj].save(os.path.join(imgs_dir,"%i_%s.png"%(ii, str(attacks[jj])) ))

        decoded_data = decode.decode_0bit(attacked_imgs, carrier, angle, model)
        for jj in range(len(attacks)):
            attack = attacks[jj].copy()
            # change params name before logging to harmonize df between attacks
            attack_name = attack.pop('attack')
            param_names = ['param%i'%kk for kk in range(len(attack.keys()))]
            attack_params = dict(zip(param_names,list(attack.values()))) 
            decoded_datum = decoded_data[jj]
            log = {
                "keyword": "evaluation",
                "img": ii,
                "attack": attack_name,
                **attack_params,
                "log10_pvalue": decoded_datum['log10_pvalue'],
                "R": decoded_datum['R'],
                "marked": decoded_datum['R']>0,
            }
            logs.append(log)
            if params.verbose>1:
                print("__log__:%s" % json.dumps(log))

    df = pd.DataFrame(logs).drop(columns='keyword')

    if params.verbose>0:
        print('\n%s'%df)
    return df


def decode_multibit_from_folder(img_dir, carrier, model, msg_type):
    """
    Args:
        img_dir: Folder containing the images to decode
        carrier (tensor of size KxD): K carriers of dimension D, each one determines a bit
        model: Neural net model to extract the features
        msg_type: Type of message to decode ('bit' or 'text')

    Returns:
        df: Dataframe with the decoded message for each image
    """
    imgs, filenames = utils_img.pil_imgs_from_folder(img_dir)
    decoded_data = decode.decode_multibit(imgs, carrier, model)
    df = pd.DataFrame(decoded_data)
    df['filename'] = filenames
    df['msg'] = df['msg'].apply(
        lambda x: ''.join(map(str,x.type(torch.int).tolist()))
    )
    if msg_type == 'text':
        df['msg'] = df['msg'].apply(
            lambda x: utils.binary_to_string(x)
        )
    return df


def evaluate_multibit_on_attacks(imgs, carrier, model, msgs_orig, params, attacks=attacks, save=True):
    """
    Args:
        imgs: Watermarked images, list of PIL Images
        carrier (tensor of size KxD): K carriers of dimension D, each one determines a bit
        model: Neural net model to extract the features
        msgs_orig (boolean tensor of size NxK): original messages
        params: Must contain verbose, output_dir 
        attacks: List of attacks to apply
        save: Whether to save instances of attacked images for the first image

    Returns:
        df: Dataframe with the decoding scores for each transformation
    """

    logs = []
    for ii, img in enumerate(tqdm(imgs)):
        
        attacked_imgs = generate_attacks(img, attacks)
        if ii==0 and save:
            imgs_dir = os.path.join(params.output_dir, 'imgs')
            for jj in range(len(attacks)):
                attacked_imgs[jj].save(os.path.join(imgs_dir,"%i_%s.png"%(ii, str(attacks[jj])) ))

        decoded_data = decode.decode_multibit(attacked_imgs, carrier, model)
        for jj in range(len(attacks)):
            attack = attacks[jj].copy()
            # change params name before logging to harmonize df between attacks
            attack_name = attack.pop('attack')
            param_names = ['param%i'%kk for kk in range(len(attack.keys()))]
            attack_params = dict(zip(param_names,list(attack.values()))) 
            decoded_datum = decoded_data[jj]
            diff = (~torch.logical_xor(msgs_orig[ii], decoded_datum['msg'])).tolist() # useful for bit accuracy metric
            log = {
                "keyword": "evaluation",
                "img": ii,
                "attack": attack_name,
                **attack_params,
                "msg_orig": msgs_orig[ii].tolist(),
                "msg_decoded": decoded_datum['msg'].tolist(),
                "bit_acc": np.sum(diff)/len(diff),
                "word_acc": int(np.sum(diff)==len(diff)),
            }
            logs.append(log)
            if params.verbose>1:
                print("__log__:%s" % json.dumps(log))

    df = pd.DataFrame(logs).drop(columns='keyword')

    if params.verbose>0:
        print('\n%s'%df)
    return df


def aggregate_df(df, params):
    """
    Reads the dataframe output by the previous function and returns average detection scores for each transformation
    """
    df['param0'] = df['param0'].fillna(-1)
    df_mean = df.groupby(['attack','param0'], as_index=False).mean().drop(columns='img')
    df_agg = df.groupby(['attack','param0'], as_index=False).agg(['mean','min','max','std']).drop(columns='img')

    if params.verbose>0:
        print('\n%s'%df_mean)
        print('\n%s'%df_agg)
    return df_agg

