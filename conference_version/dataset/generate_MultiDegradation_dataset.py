import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import random
import torch
import argparse
from tqdm import tqdm
import dataset.lowlevel_prompt_update_dataloader as lowlevel_prompt_dataloader

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='xrestormer_prompt_crossattn_base', type=str, metavar='MODEL',
                        help='Name of model to validation')
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--data_path_val', default='data/Common528/Image_clean_256x256', type=str,)
    parser.add_argument('--output_dir', default='data/Common528/MixDegradation', type=str,)
    # parser.add_argument('--data_path', default='../datasets/prompt_generalization_testset_256')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tta_option', default=0, type=int)
    parser.add_argument('--ckpt', default='./experiments/108_main_train_xrestormer_prompt_crosskvattn_wores_restoration_correct/checkpoint-15.pth')

    parser.set_defaults(autoregressive=False)
    return parser

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    dataset_val = lowlevel_prompt_dataloader.DatasetPrompt_ConstantParameter_Val(dataset_path=args.data_path_val,
                                                                input_size=args.input_size, tasks_flag=2)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        #remove output_dir
        os.system(f'rm -rf {args.output_dir}')
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'input_query_img1')):
        os.makedirs(os.path.join(args.output_dir, 'input_query_img1'))
    if not os.path.exists(os.path.join(args.output_dir, 'input_query_img2')):
        os.makedirs(os.path.join(args.output_dir, 'input_query_img2'))
    if not os.path.exists(os.path.join(args.output_dir, 'target_img1')):
        os.makedirs(os.path.join(args.output_dir, 'target_img1'))
    if not os.path.exists(os.path.join(args.output_dir, 'target_img2')):
        os.makedirs(os.path.join(args.output_dir, 'target_img2'))
    
    for idx, data in tqdm(enumerate(data_loader_val)):
        batch, deg_type = data
        input_img1 = batch['input_query_img1']
        target_img1 = batch['target_img1']
        input_img2 = batch['input_query_img2']
        target_img2 = batch['target_img2']
        
        input_img1 = input_img1[0].permute(1, 2, 0)*255.0
        input_img2 = input_img2[0].permute(1, 2, 0)*255.0
        target_img1 = target_img1[0].permute(1, 2, 0)*255.0
        target_img2 = target_img2[0].permute(1, 2, 0)*255.0
        
        #save to output_dir
        save_path_input_query_img1 = os.path.join(args.output_dir, 'input_query_img1')
        save_path_input_query_img2 = os.path.join(args.output_dir, 'input_query_img2')
        save_path_target_img1 = os.path.join(args.output_dir, 'target_img1')
        save_path_target_img2 = os.path.join(args.output_dir, 'target_img2')
        
        img_name = f'{idx}_'+deg_type[0]+'.png'
        
        cv2.imwrite(os.path.join(save_path_input_query_img1, img_name), input_img1.numpy().astype(np.uint8))
        cv2.imwrite(os.path.join(save_path_input_query_img2, img_name), input_img2.numpy().astype(np.uint8))
        cv2.imwrite(os.path.join(save_path_target_img1, img_name), target_img1.numpy().astype(np.uint8))
        cv2.imwrite(os.path.join(save_path_target_img2, img_name), target_img2.numpy().astype(np.uint8))
        
        
        
    


