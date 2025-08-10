import torch
import argparse
import numpy as np
from pathlib import Path
import os

import util.misc as misc
import torch.backends.cudnn as cudnn

import models_xrestormer_prompt_crossattn_wores
from evaluation_wo_prompt import inference_model_wo_Prompt
from evaluation_vit_prompt import inference_on_ViT_Prompt, val_on_ViT_Prompt
import dataset.lowlevel_prompt_update_dataloader as lowlevel_dataloader
import dataset.util as util
import cv2
from PIL import Image

import json
import random
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, choices=['huge', 'giant'], default='giant')
    parser.add_argument('--model', default='xrestormer_prompt_crossattn_huge', type=str, metavar='MODEL', help='Name of model to validation')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tta_option', default=0, type=int)
    parser.add_argument('--ckpt', default='ckpt/GenLV-Giant/checkpoint-17.pth')
    parser.add_argument('--prompt_input', type=str)
    parser.add_argument('--prompt_target', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', default='example/prediction.png', type=str)

    parser.set_defaults(autoregressive=False)
    return parser

def load_img(img_path):
    img = util.read_img(None, str(img_path), None)
    img = cv2.resize(np.copy(img), (256, 256), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float().cuda()
    return img
    

def main(args):
    # build the model
    model = models_xrestormer_prompt_crossattn_wores.__dict__[args.model]()
    # load model
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    model.to(torch.device(args.device))
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    img_gt1 = load_img(args.prompt_target)
    img_lq1 = load_img(args.prompt_input)
    img_lq2 = load_img(args.input_path)
    
    input_sequence = [img_lq1[None], img_gt1[None], img_lq2[None], img_lq2[None]]
    
    with torch.no_grad():
        model.eval()
        _, pred_target_img, _ = model(imgs=input_sequence, input_is_list=True)
    
    pred_target_img = torch.einsum('nchw->nhwc', pred_target_img).detach().cpu()
    pred_target_img = (torch.clip((pred_target_img[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
    pred_target_img = pred_target_img[0].numpy().astype(np.uint8)
    cv2.imwrite(args.output_path, pred_target_img)

    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    # # manually set
    # args.prompt_input = 'example/prompt_input.png'
    # args.prompt_target = 'example/prompt_target.png'
    # args.input_path = 'example/input.png'
    
    # args.model_size = 'giant'
    
    model_dict = {
        'huge': {
            'model': 'xrestormer_prompt_crossattn_huge',
            'ckpt': 'ckpt/GenLV-Huge/checkpoint-22.pth'
        },
        'giant': {
            'model': 'xrestormer_prompt_crossattn_giant',
            'ckpt': 'ckpt/GenLV-Giant/checkpoint-17.pth'
        }
    }
    
    args.model = model_dict[args.model_size]['model']
    args.ckpt = model_dict[args.model_size]['ckpt']
    
    main(args)