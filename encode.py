# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import numpy as np
import torch
from tqdm import tqdm

import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(name, model_params, **optim_params):
    """ Build optimizer from a dictionary of parameters """
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)(model_params, **optim_params)
    raise ValueError(f'Unknown optimizer "{name}"')


def build_lr_scheduler(name, optimizer, **lr_scheduler_params):
    """ Build scheduler from a dictionary of parameters """
    if hasattr(torch.optim.lr_scheduler, name):
        return getattr(torch.optim.lr_scheduler, name)(optimizer, **lr_scheduler_params)
    raise ValueError(f'Unknown LR scheduler "{name}"')


def watermark_0bit(img_loader, carrier, angle, model, transform, params):
    """
    0-bit watermarking of a batch of images.

    Args:
        img_loader: Dataloader of the images to be watermarked
        carrier (tensor of size 1xD): Hypercone direction 1xD
        angle: Angle of the hypercone
        model: Neural net model to extract the features
        transform: Differentiable augmentation with fixed output size -> 1xCxWxH
        params: Must contain optimizer, scheduler, epochs, lambda_w, lambda_i, verbose

    Returns:
        imgs: Watermarked images as a list of unnormalized (distributed around [-1, 1]) pytorch tensors
    """
    rho = 1 + np.tan(angle)**2
    ssim = utils_img.SSIMAttenuation(device=device)
    pt_imgs_out = []

    for batch_iter, (images, _) in enumerate(tqdm(img_loader)):

        # Warning for resolution
        max_res = max([img.shape[-1]*img.shape[-2] for img in images])
        if max_res > 1e6:
            print('WARNING: One or more of the images is high resolution, it can be too large to be processed by the GPU.')

        # load images
        batch_imgs_orig = [x.to(device, non_blocking=True).unsqueeze(0) for x in images] # BxCxWxH
        batch_imgs = [x.clone() for x in batch_imgs_orig] # BxCxWxH
        for i in range(len(batch_imgs)):
            batch_imgs[i].requires_grad = True
        optimizer = build_optimizer(model_params=batch_imgs, **utils.parse_params(params.optimizer))
        if params.scheduler is not None:
            scheduler = build_lr_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler))

        # optimization
        for iteration in range(params.epochs):
            # Constraints and data augmentations
            batch = []
            for ii, x in enumerate(batch_imgs):
                # concentrate changes around edges
                x = ssim.apply(x, batch_imgs_orig[ii])
                # remain within PSNR budget
                x = utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
                if ii==0:
                    aug_params = transform.sample_params(x)
                aug_img = transform(x, aug_params)
                batch.append(aug_img)
            batch = torch.cat(batch, dim=0) # BxCxWxH
            # get features
            ft = model(batch) # BxCxWxH -> BxD
            # compute losses
            dot_product = ft @ carrier.T # BxD @ Dx1 -> Bx1
            norm = torch.norm(ft, dim=-1, keepdim=True) # BxD -> Bx1
            loss_w = torch.sum(-(rho * dot_product**2 - norm**2)) # B-B -> B
            loss_i = 0
            for ii in range(len(batch_imgs)):
                loss_i += torch.norm(batch_imgs[ii] - batch_imgs_orig[ii])**2 # CxWxH -> 1
            loss = params.lambda_w*loss_w + params.lambda_i*loss_i
            # update images (gradient descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params.scheduler is not None:
                scheduler.step()
            # logs
            if params.verbose>1:
                logs = {
                    "keyword": "img_optim",
                    "batch": batch_iter,
                    "iteration": iteration,
                    "loss": loss.item(),
                    "loss_w": loss_w.item(),
                    "loss_i": loss_i.item(),
                }
                if params.verbose>2:
                    rs = rho * dot_product**2 - norm**2 # Bx1-Bx1 -> Bx1
                    cosines = torch.abs(dot_product/norm) # Bx1/Bx1 -> Bx1
                    log10_pvalues = [np.log10(utils.cosine_pvalue(cosines[ii].item(), ft.shape[-1])) for ii in range(len(batch_imgs))]
                    logs["R_avg"] = torch.mean(rs).item()
                    logs["R_min_max"] = (torch.min(rs).item(), torch.max(rs).item())
                    logs["log10_pvalue_avg"] = np.mean(log10_pvalues)
                    logs["log10_pvalue_min_max"] = (np.amin(log10_pvalues), np.amax(log10_pvalues))
                print("__log__:%s" % json.dumps(logs))

        # post process and store
        for ii,x in enumerate(batch_imgs):
            x = ssim.apply(x, batch_imgs_orig[ii])
            x = utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
            x = utils_img.round_pixel(x)
            # x = utils_img.project_linf(x, batch_imgs_orig[ii], params.linf_radius)
            pt_imgs_out.append(x.squeeze(0).detach().cpu())

    return pt_imgs_out # [CxW1xH1, ..., CxWnxHn] 


def watermark_multibit(img_loader, msgs, carrier, model, transform, params):
    """
    multi-bit watermarking of a batch of images.

    Args:
        img_loader: Dataloader of the images to be watermarked
        msgs (boolean tensor of size NxK): messages to be encoded in the N images   
        carrier (tensor of size KxD): K carriers of dimension D, each one determines a bit
        model: Neural net model to extract the features
        transform: Differentiable augmentation with fixed output size -> 1xCxWxH
        params: Must contain batch_size, optimizer, scheduler, epochs, lambda_w, lambda_i, verbose

    Returns:
        imgs: Watermarked images as a list of unnormalized (distributed around [-1, 1]) pytorch tensors
    """

    def message_loss(ft, carrier, msgs, m=5):
        dot_products = ft @ carrier.T # BxD @ DxK -> BxK
        msg_signs = 2*msgs.type(torch.float)-1 # BxK
        return torch.sum(torch.clamp(m-dot_products*msg_signs, min=0)) / msg_signs.size(-1)

    ssim = utils_img.SSIMAttenuation(device=device)
    pt_imgs_out = []

    for batch_iter, (images, _) in enumerate(tqdm(img_loader)):

        # Warning for resolution
        max_res = max([img.shape[-1]*img.shape[-2] for img in images])
        if max_res > 1e6:
            print('WARNING: One or more of the images is high resolution, it can be too large to be processed by the GPU.')

        # load images
        batch_imgs_orig = [x.to(device, non_blocking=True).unsqueeze(0) for x in images] # BxCxWxH
        batch_imgs = [x.clone() for x in batch_imgs_orig] # BxCxWxH
        for i in range(len(batch_imgs)):
            batch_imgs[i].requires_grad = True
        N = len(img_loader.dataset)
        B = params.batch_size
        batch_msgs = msgs[batch_iter*B : min((batch_iter+1)*B, N)].to(device, non_blocking=True)
        optimizer = build_optimizer(model_params=batch_imgs, **utils.parse_params(params.optimizer))
        if params.scheduler is not None:
            scheduler = build_lr_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler))

        # optimization
        for iteration in range(params.epochs):
            # Constraints and data augmentations
            batch = []
            for ii, x in enumerate(batch_imgs):
                x = ssim.apply(x, batch_imgs_orig[ii])
                x = utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
                if ii==0:
                    aug_params = transform.sample_params(x)
                aug_img = transform(x, aug_params)
                batch.append(aug_img)
            batch = torch.cat(batch, dim=0) # BxCxWxH
            # get features
            ft = model(batch) # BxCxWxH -> BxD
            # compute losses
            loss_w = message_loss(ft, carrier, batch_msgs)
            loss_i = 0
            for ii in range(len(batch_imgs)):
                loss_i += torch.norm(batch_imgs[ii] - batch_imgs_orig[ii])**2 # CxWxH -> 1
            loss = params.lambda_w*loss_w + params.lambda_i*loss_i
            # update images (gradient descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params.scheduler is not None:
                scheduler.step()
            # logs
            if params.verbose>1:
                logs = {
                    "keyword": "img_optim",
                    "batch": batch_iter,
                    "iteration": iteration,
                    "loss": loss.item(),
                    "loss_w": loss_w.item(),
                    "loss_i": loss_i.item(),
                }
                if params.verbose>2:
                    dot_product = (ft @ carrier.T) # BxD @ DxK -> BxK
                    decoded_msgs = torch.sign(dot_product) > 0 # BxK -> BxK
                    diff = (~torch.logical_xor(batch_msgs, decoded_msgs)) # BxK -> BxK
                    bit_accs = torch.sum(diff, dim=-1)/diff.shape[-1] # BxK -> B
                    logs["bit_acc_avg"] = torch.mean(bit_accs).item()
                    logs["R_min_max"] = (torch.min(bit_accs).item(), torch.max(bit_accs).item())
                print("__log__:%s" % json.dumps(logs))

        # post process and store
        for ii,x in enumerate(batch_imgs):
            x = ssim.apply(x, batch_imgs_orig[ii])
            x = utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
            x = utils_img.round_pixel(x)
            # x = utils_img.project_linf(x, batch_imgs_orig[ii], params.linf_radius)
            pt_imgs_out.append(x.squeeze(0).detach().cpu())

    return pt_imgs_out # [CxW1xH1, ..., CxWnxHn] 

