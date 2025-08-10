import os
from tqdm import tqdm

import cv2
import numpy as np
import torch


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

def inference_on_synthetic_data(model, data_loader, save_path, task_type, save_prompt=False):
    for batch, deg_info in data_loader:
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
        
        save_img_path = os.path.join(save_path, 'Synthetic_Test', task_type)
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        
        idx = os.path.basename(batch['input_query_img2_path'][0])[:3]
        if save_prompt:
            prompt_img = create_2_grid_from_np_images(input_img1, target_img1)
            cv2.imwrite(os.path.join(save_img_path, '{}_prompt_{}.png'.format(idx, deg_info[0])), prompt_img)

        # if target_img2_flag:
        #     output_img = create_3_grid_from_np_images(input_img2, pred_target_img, target_img2)
        #     cv2.imwrite(os.path.join(save_img_path, '{}_input_{}.png'.format(idx, deg_info[0])), output_img)
        # else:
        output_img = create_2_grid_from_np_images(input_img2, pred_target_img)
        cv2.imwrite(os.path.join(save_img_path, '{}_{}.png'.format(idx, deg_info[0])), output_img)


def inference_on_customize_prompt(model, data_loader, save_path, save_prompt=False):
    for batch, task_info in tqdm(data_loader, desc=data_loader.dataset.task_info):
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
        
        save_img_path = os.path.join(save_path, task_info[0].split('-')[0], task_info[0].split('-')[1])
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        
        if save_prompt:
            prompt_img = create_2_grid_from_np_images(input_img1, target_img1)
            cv2.imwrite(os.path.join(save_img_path, '00000_prompt.png'), prompt_img)
            save_prompt = False

        if target_img2_flag:
            output_img = create_3_grid_from_np_images(input_img2, pred_target_img, target_img2)
            cv2.imwrite(os.path.join(save_img_path, os.path.basename(batch['input_query_img2_path'][0])), output_img)
        else:
            output_img = create_2_grid_from_np_images(input_img2, pred_target_img)
            cv2.imwrite(os.path.join(save_img_path, os.path.basename(batch['input_query_img2_path'][0])), output_img)
            

def inference_on_individual_data(model, data_loader, save_path, save_prompt=False):
    for batch, task_info in data_loader:
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
        
        if not isinstance(task_info[0], list):
            save_img_path = os.path.join(save_path, task_info[0].split('=')[0])
        else:
            save_img_path = os.path.join(save_path, task_info[0][0])
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        
        if save_prompt:
            if not isinstance(task_info[0], list):
                prompt_img = create_2_grid_from_np_images(input_img1, target_img1)
                cv2.imwrite(os.path.join(save_img_path, '00000_prompt_{}.png'.format(task_info[0].split('=')[1])), prompt_img)
                save_prompt = False
            else:
                prompt_img = create_2_grid_from_np_images(input_img1, target_img1)
                # style = os.path.basename(batch['input_query_img2_path'][0])[0]
                # cv2.imwrite(os.path.join(save_img_path, '00000_prompt_{}.png'.format(task_info[0][int(style)])), prompt_img)
                cv2.imwrite(os.path.join(save_img_path, '{}_prompt.png'.format(os.path.basename(batch['input_query_img2_path'][0]))), prompt_img)
                
        if target_img2_flag:
            output_img = create_3_grid_from_np_images(input_img2, pred_target_img, target_img2)
            cv2.imwrite(os.path.join(save_img_path, os.path.basename(batch['input_query_img2_path'][0])), output_img)
        else:
            output_img = create_2_grid_from_np_images(input_img2, pred_target_img)
            cv2.imwrite(os.path.join(save_img_path, os.path.basename(batch['input_query_img2_path'][0])), output_img)