import torch
import numpy as np
import cv2
import os
import torch.distributed as dist
from tqdm import tqdm

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def create_2_grid_from_np_images(input_img, pred_img):
    canvas = np.ones((pred_img.shape[0], 2 * pred_img.shape[1], pred_img.shape[2]))
    canvas[:input_img.shape[0], :input_img.shape[1], :] = input_img
    canvas[:input_img.shape[0], input_img.shape[1]:, :] = pred_img
    return canvas

def create_3_grid_from_np_images(input_img, pred_img, target_img):
    canvas = np.ones((target_img.shape[0], 3 * target_img.shape[1], target_img.shape[2]))
    canvas[:input_img.shape[0], :input_img.shape[1], :] = input_img
    canvas[:input_img.shape[0], input_img.shape[1]:2 * input_img.shape[1], :] = pred_img
    canvas[:input_img.shape[0], 2 * input_img.shape[1]:, :] = target_img
    return canvas

def val_wo_Prompt(model, data_loader_val, epoch, save_path, mode, patch_size):
    if is_main_process():
        val_model_wo_Prompt(model, data_loader_val, epoch, save_path, mode, patch_size)

def val_model_wo_Prompt(model, data_loader_val, epoch, save_path, mode='val_pretrain', patch_size=16):
    idx = 0
    for batch, deg_type in data_loader_val:
        idx += 1
        input_img = batch['input_query_img'].cuda()
        if batch['target_img'][0] != 'None':
            target_img = batch['target_img'].cuda()
            target_img_flag = True
        else:
            target_img = input_img
            target_img_flag = False

        input_sequence = [input_img, target_img]
        model.eval()
        _, pred_target_img, _ = model(imgs=input_sequence, input_is_list=True)

        input_img = torch.einsum('nchw->nhwc', input_img)
        target_img = torch.einsum('nchw->nhwc', target_img)
        pred_target_img = torch.einsum('nchw->nhwc', pred_target_img).detach().cpu()

        input_img = (torch.clip((input_img[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_img = (torch.clip((target_img[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_img = (torch.clip((pred_target_img[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        input_img = input_img[0].numpy().astype(np.uint8)
        target_img = target_img[0].numpy().astype(np.uint8)
        pred_target_img = pred_target_img[0].numpy().astype(np.uint8)
        
        save_path_img = os.path.join(save_path, 'vis', mode, 'Epoch{}'.format(epoch))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)

        if target_img_flag:
            output_img = create_3_grid_from_np_images(input_img, pred_target_img, target_img)
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img_{}.png'.format(idx, deg_type[0])), output_img)
        else:
            output_img = create_2_grid_from_np_images(input_img, pred_target_img)
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img_{}.png'.format(idx, deg_type[0])), output_img)
    
def inference_model_wo_Prompt(model, data_loader_val, save_path, deg_type=''):
    idx = 0
    print('Inference on {}'.format(deg_type))
    for batch, _ in tqdm(data_loader_val):
        idx += 1
        input_img = batch['input_query_img'].cuda()
        if batch['target_img'][0] != 'None':
            target_img = batch['target_img'].cuda()
            target_img_flag = True
        else:
            target_img = input_img
            target_img_flag = False

        input_sequence = [input_img, target_img]
        model.eval()
        _, pred_target_img, _ = model(imgs=input_sequence, input_is_list=True)

        input_img = torch.einsum('nchw->nhwc', input_img)
        target_img = torch.einsum('nchw->nhwc', target_img)
        pred_target_img = torch.einsum('nchw->nhwc', pred_target_img).detach().cpu()

        input_img = (torch.clip((input_img[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_img = (torch.clip((target_img[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_img = (torch.clip((pred_target_img[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        input_img = input_img[0].numpy().astype(np.uint8)
        target_img = target_img[0].numpy().astype(np.uint8)
        pred_target_img = pred_target_img[0].numpy().astype(np.uint8)
        
        output_path = os.path.join(save_path, deg_type)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if target_img_flag:
            output_img = create_3_grid_from_np_images(input_img, pred_target_img, target_img)
            cv2.imwrite(os.path.join(output_path, 'TestID{:04d}_input_img_{}.png'.format(idx, deg_type[0])), output_img)
        else:
            output_img = create_2_grid_from_np_images(input_img, pred_target_img)
            cv2.imwrite(os.path.join(output_path, 'TestID{:04d}_input_img_{}.png'.format(idx, deg_type[0])), output_img)