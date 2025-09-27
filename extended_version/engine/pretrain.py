# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import util.misc as misc
import util.lr_sched as lr_sched

from util.loss import GANLoss

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_size=1):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_i = iter(data_loader)
    for data_iter_step in metric_logger.log_every(range(epoch_size), print_freq, header):
        (batch, _) = next(data_loader_i)
        # we use a per iteration (instead of per epoch) lr scheduler
        if isinstance(batch, tuple):
            samples, visual_tokens = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = visual_tokens.to(device, non_blocking=True)
        else: # hack for consistency
            samples = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = samples

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Non-MAE based
        with torch.cuda.amp.autocast():
            loss_dict, _, _ = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        # MAE based
        # with torch.cuda.amp.autocast():
        #     loss_dict = model(samples, visual_tokens)[0]

        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # loss_scaler(loss, optimizer, clip_grad=0.5, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_mae(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_size=1):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_i = iter(data_loader)
    for data_iter_step in metric_logger.log_every(range(epoch_size), print_freq, header):
        (batch, _) = next(data_loader_i)
        # we use a per iteration (instead of per epoch) lr scheduler
        if isinstance(batch, tuple):
            samples, visual_tokens = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = visual_tokens.to(device, non_blocking=True)
        else: # hack for consistency
            samples = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = samples

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Non-MAE based
        # with torch.cuda.amp.autocast():
        #     loss_dict, _, _ = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        # MAE based
        with torch.cuda.amp.autocast():
            loss_dict = model(samples, visual_tokens)[0]

        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # loss_scaler(loss, optimizer, clip_grad=0.5, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

ls_gan_loss = GANLoss('lsgan', 1.0, 0.0)
L1_loss = nn.L1Loss()

def adversarial_train_one_epoch(model: torch.nn.Module, model_D: torch.nn.Module, model_F: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, optimizer_D: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, loss_scaler_D,
                    log_writer=None,
                    args=None,
                    epoch_size=1,
                    l_pix_w=0.1,
                    l_fea_w=1,
                    l_gan_w=5e-3,
                    lr_decay=True):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    optimizer_D.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_i = iter(data_loader)
    for data_iter_step in metric_logger.log_every(range(epoch_size), print_freq, header):
        (batch, _) = next(data_loader_i)
        # we use a per iteration (instead of per epoch) lr scheduler
        if isinstance(batch, tuple):
            samples, visual_tokens = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = visual_tokens.to(device, non_blocking=True)
        else: # hack for consistency
            samples = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = samples

        if data_iter_step % accum_iter == 0:
            if lr_decay:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            else:
                pass
        with torch.cuda.amp.autocast():
            loss_dict, pred, _ = model(samples, visual_tokens, mask_ratio=args.mask_ratio)
        
        # G
        for p in model_D.parameters():
            p.requires_grad = False
        # pix loss (in loss_dict)
        loss_dict['pix_loss'] = l_pix_w * loss_dict['pix_loss']
        
        # perceptual loss
        target_img = visual_tokens[:, 3, :, :, :]
        pred_img = pred.float()
        
        real_fea = model_F(target_img).detach()
        fake_fea = model_F(pred_img)
        l_g_fea = l_fea_w * L1_loss(real_fea, fake_fea)
        
        loss_dict['percep_loss'] = l_g_fea
        
        # adversarial loss G
        pred_g_fake = model_D(pred_img)
        l_g_gan = l_gan_w * ls_gan_loss(pred_g_fake, True)
        
        loss_dict['gan_G'] = l_g_gan
        
        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # D
        for p in model_D.parameters():
            p.requires_grad = True

        loss_D_dict = {}
        l_d_total = 0
        pred_d_real = model_D(target_img)
        pred_d_fake = model_D(pred_img.detach())  # detach to avoid BP to G

        l_d_real = ls_gan_loss(pred_d_real, True)
        l_d_fake = ls_gan_loss(pred_d_fake, False)
        l_d_total = l_d_real + l_d_fake
        
        loss_D_dict['gan_D'] = l_d_total

        loss_D = torch.stack([loss_D_dict[l] for l in loss_D_dict if 'unscaled' not in l]).sum()
        loss_value_D = loss_D.item()

        if not math.isfinite(loss_value_D):
            print("Loss is {}, stopping training".format(loss_value_D))
            sys.exit(1)

        loss_D /= accum_iter
        loss_scaler_D(loss_D, optimizer_D, parameters=model_D.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_D.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{k: v.item() for k, v in loss_D_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_value_reduce_D = misc.all_reduce_mean(loss_value_D)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss_G', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_D', loss_value_reduce_D, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model, data_loader, device, epoch, log_writer, args):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, visual_tokens = batch
        samples = samples.to(device, non_blocking=True)
        visual_tokens = visual_tokens.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
        # with torch.cuda.amp.autocast(enabled=False):
            loss_dict, _, _ = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats for val:", metric_logger)
    return {'val_' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
