# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
import torch
from torchvision.transforms import ToPILImage

import data_augmentation
import encode
import evaluate
import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--data_dir", type=str, default="input/", help="Folder directory (Default: /input)")
    aa("--carrier_dir", type=str, default="carriers/", help="Directions of the latent space in which the watermark is embedded (Default: /carriers)")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--save_images", type=utils.bool_inst, default=True, help="Whether to save watermarked images (Default: True)")
    aa("--evaluate", type=utils.bool_inst, default=True, help="Whether to evaluate the detector (Default: True)")
    aa("--decode_only", type=utils.bool_inst, default=False, help="To decode only watermarked images (Default: False)")
    aa("--verbose", type=int, default=1)

    group = parser.add_argument_group('Marking parameters')
    aa("--target_psnr", type=float, default=42.0, help="Target PSNR value in dB. (Default: 42 dB)")
    aa("--target_fpr", type=float, default=1e-6, help="Target FPR of the dectector. (Default: 1e-6)")

    group = parser.add_argument_group('Neural-Network parameters')
    aa("--model_name", type=str, default='resnet50', help="Marking network architecture. See https://pytorch.org/vision/stable/models.html and https://rwightman.github.io/pytorch-image-models/models/ (Default: resnet50)")
    aa("--model_path", type=str, default="models/dino_r50_plus.pth", help="Path to the model (Default: /models/dino_r50_plus.pth)")
    aa("--normlayer_path", type=str, default="normlayers/out2048_yfcc_orig.pth", help="Path to the normalization layer (Default: /normlayers/out2048.pth)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--epochs", type=int, default=100, help="Number of epochs for image optimization. (Default: 100)")
    aa("--data_augmentation", type=str, default="all", choices=["none", "all"], help="Type of data augmentation to use at marking time. (Default: All)")
    aa("--optimizer", type=str, default="Adam,lr=0.01", help="Optimizer to use. (Default: Adam,lr=0.01)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--batch_size", type=int, default=1, help="Batch size for marking. (Default: 128)")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=1.0, help="Weight of the image loss. (Default: 1.0)")

    return parser


def main(params):
    # Set seeds for reproductibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Loads backbone and normalization layer
    if params.verbose > 0:
        print('>>> Building backbone and normalization layer...')
    backbone = utils.build_backbone(path=params.model_path, name=params.model_name)
    normlayer = utils.load_normalization_layer(path=params.normlayer_path)
    model = utils.NormLayerWrapper(backbone, normlayer)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Load or generate carrier and angle
    if not os.path.exists(params.carrier_dir):
        os.makedirs(params.carrier_dir, exist_ok=True)
    D = model(torch.zeros((1,3,224,224)).to(device)).size(-1)
    carrier_path = os.path.join(params.carrier_dir,'carrier_%i_%i.pth'%(1, D))
    if os.path.exists(carrier_path):
        if params.verbose > 0:
            print('>>> Loading carrier from %s' % carrier_path)
        carrier = torch.load(carrier_path)
        assert D == carrier.shape[1]
    else:
        if params.verbose > 0:
            print('>>> Generating carrier into %s' % carrier_path)
        carrier = utils.generate_carriers(1, D, output_fpath=carrier_path)
    carrier = carrier.to(device, non_blocking=True) # direction vector of the hypercone
    angle = utils.pvalue_angle(dim=D, k=1, proba=params.target_fpr) # angle of the hypercone

    # Decode only
    if params.decode_only:
        if params.verbose > 0:
            print('>>> Decoding watermarks...')
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir, exist_ok=True)
        df = evaluate.decode_0bit_from_folder(params.data_dir, carrier, angle, model)
        df_path = os.path.join(params.output_dir,'decodings.csv')
        df.to_csv(df_path, index=False)
        if params.verbose > 0:
            print('Results saved in %s'%df_path)

    else:
        # Construct data augmentation
        if params.data_augmentation == 'all':
            data_aug = data_augmentation.All()
        elif params.data_augmentation == 'none':
            data_aug = data_augmentation.DifferentiableDataAugmentation()

        # Load images
        if params.verbose > 0:
            print('>>> Loading images from %s...'%params.data_dir)
        dataloader = utils_img.get_dataloader(params.data_dir, batch_size=params.batch_size)

        # Marking
        if params.verbose > 0:
            print('>>> Marking images...')
        pt_imgs_out = encode.watermark_0bit(dataloader, carrier, angle, model, data_aug, params)
        imgs_out = [ToPILImage()(utils_img.unnormalize_img(pt_img).squeeze(0)) for pt_img in pt_imgs_out] 
        
        # Evaluate
        if params.evaluate:
            if params.verbose > 0:
                print('>>> Evaluating watermarks...')
            if not os.path.exists(params.output_dir):
                os.makedirs(params.output_dir, exist_ok=True)
            imgs_dir = os.path.join(params.output_dir, 'imgs')
            if not os.path.exists(imgs_dir):
                os.mkdir(imgs_dir)
            df = evaluate.evaluate_0bit_on_attacks(imgs_out, carrier, angle, model, params)
            df_agg = evaluate.aggregate_df(df, params)
            df_path = os.path.join(params.output_dir,'df.csv')
            df_agg_path = os.path.join(params.output_dir,'df_agg.csv')
            df.to_csv(df_path, index=False)
            df_agg.to_csv(df_agg_path)
            if params.verbose > 0:
                print('Results saved in %s'%df_path)
        
        # Save
        if params.save_images:
            if not os.path.exists(params.output_dir):
                os.makedirs(params.output_dir, exist_ok=True)
            imgs_dir = os.path.join(params.output_dir, 'imgs')
            if params.verbose > 0:
                print('>>> Saving images into %s...'%imgs_dir)
            if not os.path.exists(imgs_dir):
                os.mkdir(imgs_dir)
            for ii, img_out in enumerate(imgs_out):
                img_out.save(os.path.join(imgs_dir, '%i_out.png'%ii))
        


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
