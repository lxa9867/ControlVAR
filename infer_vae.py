import os
import argparse
import logging
import math
import numpy as np
from time import time
from datetime import datetime
from tqdm.auto import tqdm
import wandb
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

from transformers import get_scheduler

from datasets import create_dataset
from models import VQVAE, VisualProgressAutoreg
from utils.wandb import CustomWandbTracker
from ruamel.yaml import YAML

def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument("--config", type=str, default='configs/train_var_ImageNet_local.yaml', help="config file used to specify parameters")
    parser.add_argument("--device", type=str, default='hpu', help="random seed")

    # data
    parser.add_argument("--data", type=str, default=None, help="data")
    parser.add_argument("--data_dir", type=str, default='imagenet/train', help="data folder")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="dataset name")
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")

    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help='lr scheduler')
    parser.add_argument("--lr_warmup_steps", type=float, default=0.03, help="warmup steps")
    parser.add_argument("--log_interval", type=int, default=100, help='log interval for steps')
    parser.add_argument("--val_interval", type=int, default=1, help='validation interval for epochs')
    parser.add_argument("--save_interval", type=str, default='5000', help='save interval')
    parser.add_argument("--mixed_precision", type=str, default='no', help='mixed precision',
                        choices=['no', 'fp16', 'bf16', 'fp8'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')

    # vqvae
    parser.add_argument("--vocab_size", type=int, default=4096, nargs='+', help="codebook size")
    parser.add_argument("--z_channels", type=int, default=32, help="latent size of vqvae")
    parser.add_argument("--ch", type=int, default=160, help="channel size of vqvae")
    parser.add_argument("--vqvae_pretrained_path", type=str, default='pretrained/vae_ch160v4096z32.pth',
                        help="vqvae pretrained path")

    # vpq model
    parser.add_argument("--v_patch_nums", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
                        help="number of patch numbers of each scale")
    parser.add_argument("--v_patch_layers", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
                        help="index of layers for predicting each scale")
    parser.add_argument("--depth", type=int, default=16, help="depth of vpq model")
    parser.add_argument("--embed_dim", type=int, default=1024, help="embedding dimension of vpq model")
    parser.add_argument("--num_heads", type=int, default=16, help="number of heads of vpq model")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="mlp ratio of vpq model")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="drop rate of vpq model")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, help="attn drop rate of vpq model")
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="drop path rate of vpq model")

    # condition model
    parser.add_argument("--condition_model", type=str, default="class_embedder", help="condition model")
    parser.add_argument("--num_classes", type=int, default=1000, help="number of classes for condition model")
    parser.add_argument("--cond_drop_rate", type=float, default=0.1, help="drop rate of condition model")

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # fFirst parse of command-line args to check for config file
    args = parser.parse_args()

    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml = YAML(typ='safe')
            with open(args.config, 'r', encoding='utf-8') as file:
                config_args = yaml.load(file)
            parser.set_defaults(**config_args)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()

    return args

def infer_vae(args):
    wandb.init(project="VPA")
    device = torch.device(args.device)
    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore

    vqvae = VQVAE(vocab_size=args.vocab_size, z_channels=args.z_channels, ch=args.ch, test_mode=True,
                  share_quant_resi=4, v_patch_nums=args.v_patch_nums).to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    if args.vqvae_pretrained_path is not None:
        vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path))

    for batch_idx, batch in enumerate(dataloader):
        images, conditions = batch
        # forward to get input ids
        with torch.no_grad():
            ms_imgs = vqvae.visualize_ms_f(images.to(device))  # [[b, 3, h, w],]
        ms_imgs.append(images)
        print(batch_idx)
        if args.device == 'hpu':
            htcore.mark_step()
        wandb.log({f"reconstruction from ms features": [wandb.Image(ms_imgs[i][0]) for i in range(len(ms_imgs))]}, step=batch_idx)



if __name__ == '__main__':
    args = parse_args()
    # Setup accelerator:
    if args.run_name is None:
        model_name = f'vqvae_ch{args.ch}v{args.vocab_size}z{args.z_channels}_vpa_d{args.depth}e{args.embed_dim}h{args.num_heads}_{args.dataset_name}_ep{args.num_epochs}_bs{args.batch_size}'
    else:
        model_name = args.run_name
    args.model_name = model_name
    timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.project_dir = f"{args.output_dir}/{timestamp}-{model_name}"  # Create an experiment folder
    os.makedirs(args.project_dir, exist_ok=True)

    # create dataset
    dataset = create_dataset(args.dataset_name, args)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
    # Calculate total batch size

    infer_vae(args)