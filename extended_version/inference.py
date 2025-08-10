import argparse
import os.path as osp
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import dataset.util
import dataset.genlv_100_test_dataloader as test_dataloader
import models.xrestormer_prompt_crossattn_wores as xrestormer_prompt_crossattn_wores
import util.misc as misc
from evaluation.inference_with_prompt import *


def get_args_parser():
    parser = argparse.ArgumentParser('GenLV100', add_help=False)
    parser.add_argument('--model', default='xrestormer_prompt_crossattn_huge', type=str, metavar='MODEL', help='Name of model to validation')
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tta_option', default=0, type=int)
    parser.add_argument('--ckpt', default='ckpt/huge-checkpoint-49.pth')
    parser.add_argument('--model_size', type=str, choices=['base', 'large', 'huge'], default='huge')
    
    parser.add_argument('--prompt_input', type=str)
    parser.add_argument('--prompt_target', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--target_path', default=None, type=str)

    parser.set_defaults(autoregressive=False)
    return parser

    
def read_img(img_path):
    img = dataset.util.read_img(None, img_path, None)
    H, W, _ = img.shape
    if H < 256 or W < 256:
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    else: 
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    
    return img.unsqueeze(0).cuda()


def inference_single(model, save_dir, prompt_inp, prompt_tgt, inp, tgt=None, save_prompt=False):
    input_img1 = prompt_inp
    target_img1 = prompt_tgt
    input_img2 = inp
    if tgt is not None:
        target_img2 = tgt
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
    
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    if save_prompt:
        prompt_img = create_2_grid_from_np_images(input_img1, target_img1)
        cv2.imwrite(osp.join(save_dir, 'prompt.png'), prompt_img)

    pred_name = "prediction.png"
    if target_img2_flag:
        output_img = create_3_grid_from_np_images(input_img2, pred_target_img, target_img2)
    else:
        output_img = create_2_grid_from_np_images(input_img2, pred_target_img)
    cv2.imwrite(osp.join(save_dir, pred_name), output_img)


def main(args):
    # build the model
    model = xrestormer_prompt_crossattn_wores.__dict__[args.model]()
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
    
    prompt_inp = read_img(args.prompt_input)
    prompt_tgt = read_img(args.prompt_target)
    inp = read_img(args.input_path)
    if args.target_path is not None:
        tgt = read_img(args.target_path)
    else:
        tgt = None
    save_dir = args.save_dir
    inference_single(model, save_dir, prompt_inp, prompt_tgt, inp, tgt, save_prompt=True)

        
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    # # manually set
    # args.prompt_input = 'example/prompt_input.png'
    # args.prompt_target = 'example/prompt_target.png'
    # args.input_path = 'example/input.png'
    # args.target_path = 'example/target.png'  # optional
    
    # args.model_size = 'huge'
    
    model_dict = {
        'base': {
            'model': 'xrestormer_prompt_crossattn_base',
            'ckpt': 'ckpt/base-checkpoint-49.pth'
        },
        'large': {
            'model': 'xrestormer_prompt_crossattn_large',
            'ckpt': 'ckpt/large-checkpoint-49.pth'
        },
        'huge': {
            'model': 'xrestormer_prompt_crossattn_huge',
            'ckpt': 'ckpt/huge-checkpoint-49.pth'
        }
    }
    
    args.model = model_dict[args.model_size]['model']
    args.ckpt = model_dict[args.model_size]['ckpt']
    
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    main(args)