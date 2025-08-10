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

def val_on_Prompt(model, data_loader_val, epoch, save_path, mode, patch_size=None):
    if is_main_process():
        val_model_Prompt(model, data_loader_val, epoch, save_path, mode, patch_size=None)

def val_model_Prompt(model, data_loader_val, epoch, save_path, mode='val_pretrain', patch_size=None):
    idx = 0
    for batch, deg_type in data_loader_val:
        idx += 1
        input_img1 = batch['input_query_img1'].cuda()
        target_img1 = batch['target_img1'].cuda()
        input_img2 = batch['input_query_img2'].cuda()
        if batch['target_img2'][0] != 'None':
            target_img2 = batch['target_img2'].cuda()
            target_img2_flag = True
        else:
            target_img2 = input_img2
            target_img2_flag = False

        input_sequence = [input_img1, target_img1, input_img2, target_img2]
        with torch.no_grad():
            model.eval()
            _, pred_target_img, _ = model(imgs=input_sequence, input_is_list=True)

        input_img1 = torch.einsum('nchw->nhwc', input_img1)
        input_img2 = torch.einsum('nchw->nhwc', input_img2)
        target_img1 = torch.einsum('nchw->nhwc', target_img1)
        target_img2 = torch.einsum('nchw->nhwc', target_img2)
        pred_target_img = torch.einsum('nchw->nhwc', pred_target_img).detach().cpu()

        input_img1 = (torch.clip((input_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        input_img2 = (torch.clip((input_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_img1 = (torch.clip((target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_img2 = (torch.clip((target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_img = (torch.clip((pred_target_img[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        input_img1 = input_img1[0].numpy().astype(np.uint8)
        input_img2 = input_img2[0].numpy().astype(np.uint8)
        target_img1 = target_img1[0].numpy().astype(np.uint8)
        target_img2 = target_img2[0].numpy().astype(np.uint8)
        pred_target_img = pred_target_img[0].numpy().astype(np.uint8)
        
        save_path_img = os.path.join(save_path, 'vis', mode, 'Epoch{}'.format(epoch))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        prompt_img = create_2_grid_from_np_images(input_img1, target_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_prompt_{}.png'.format(idx, deg_type[0])), prompt_img)

        if target_img2_flag:
            output_img = create_3_grid_from_np_images(input_img2, pred_target_img, target_img2)
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img_{}.png'.format(idx, deg_type[0])), output_img)
        else:
            output_img = create_2_grid_from_np_images(input_img2, pred_target_img)
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img_{}.png'.format(idx, deg_type[0])), output_img)

        # cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img_{}.png'.format(idx, deg_type[0])), input_img)
        # if target_img_flag:
        #     cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img.png'.format(idx)), target_img)
        # cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img.png'.format(idx)), pred_target_img)

def inference_on_Prompt(model, data_loader_val, prompt_idx, save_path, mode, patch_size, save_prompt):
    if is_main_process():
        inference_model_Prompt(model, data_loader_val, prompt_idx, save_path, mode, patch_size, save_prompt)

def inference_model_Prompt(model, data_loader_val, prompt_idx, save_path, mode='val_pretrain', patch_size=16, save_prompt=False):
    idx = 0
    print("Inference on {}".format(mode))
    for batch, deg_type in tqdm(data_loader_val):
        idx += 1
        input_img1 = batch['input_query_img1'].cuda()
        target_img1 = batch['target_img1'].cuda()
        input_img2 = batch['input_query_img2'].cuda()
        if batch['target_img2'][0] != 'None':
            target_img2 = batch['target_img2'].cuda()
            target_img2_flag = True
        else:
            target_img2 = input_img2
            target_img2_flag = False

        input_sequence = [input_img1, target_img1, input_img2, target_img2]
        with torch.no_grad():
            model.eval()
            _, pred_target_img, _ = model(imgs=input_sequence, input_is_list=True)

        input_img1 = torch.einsum('nchw->nhwc', input_img1)
        input_img2 = torch.einsum('nchw->nhwc', input_img2)
        target_img1 = torch.einsum('nchw->nhwc', target_img1)
        target_img2 = torch.einsum('nchw->nhwc', target_img2)
        pred_target_img = torch.einsum('nchw->nhwc', pred_target_img).detach().cpu()

        input_img1 = (torch.clip((input_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        input_img2 = (torch.clip((input_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_img1 = (torch.clip((target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_img2 = (torch.clip((target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_img = (torch.clip((pred_target_img[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        input_img1 = input_img1[0].numpy().astype(np.uint8)
        input_img2 = input_img2[0].numpy().astype(np.uint8)
        target_img1 = target_img1[0].numpy().astype(np.uint8)
        target_img2 = target_img2[0].numpy().astype(np.uint8)
        pred_target_img = pred_target_img[0].numpy().astype(np.uint8)
        
        save_path_img = os.path.join(save_path, mode)
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        if len(deg_type) > 1:
            deg_type, prompt_deg_type = deg_type
            deg_type = deg_type[0]
            prompt_deg_type = prompt_deg_type[0]
        else:
            deg_type = deg_type[0]
            prompt_deg_type = deg_type
             
        if save_prompt:
            prompt_img = create_2_grid_from_np_images(input_img1, target_img1)
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_prompt_{}.png'.format(idx, prompt_deg_type)), prompt_img)

        if target_img2_flag:
            output_img = create_3_grid_from_np_images(input_img2, pred_target_img, target_img2)
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img_{}.png'.format(idx, deg_type)), output_img)
        else:
            output_img = create_2_grid_from_np_images(input_img2, pred_target_img)
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img_{}.png'.format(idx, deg_type)), output_img)