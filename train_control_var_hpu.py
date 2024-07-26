import os
import argparse
import math
import random

import numpy as np
from itertools import chain
from time import time
from datetime import datetime
from tqdm.auto import tqdm
import wandb
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from transformers import get_scheduler

from datasets import create_dataset
from models import VQVAE, VisualProgressAutoreg, VAR, build_var, ControlVAR, build_control_var
from ruamel.yaml import YAML
from utils import seed_everything, filter_params, lr_wd_annealing

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, tuners

device = torch.device('hpu')

def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument("--config", type=str, default='configs/train_mask_var_ImageNetS_local.yaml', help="config file used to specify parameters")

    # data
    parser.add_argument("--data", type=str, default=None, help="data")
    parser.add_argument("--data_dir", type=str, default='/voyager/ImageNet2012', help="data folder")
    parser.add_argument("--dataset_name", type=str, default="imagenetM", help="dataset name")
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--batch_size", type=int, default=8, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="batch size")

    # training
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='lin0', help='lr scheduler')
    parser.add_argument("--log_interval", type=int, default=500, help='log interval for steps')
    parser.add_argument("--val_interval", type=int, default=1, help='validation interval for epochs')
    parser.add_argument("--save_interval", type=str, default='3000', help='save interval')
    parser.add_argument("--mixed_precision", type=str, default='bf16', help='mixed precision', choices=['no', 'fp16', 'bf16', 'fp8'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')
    parser.add_argument("--lora", type=bool, default=False, help='use lora to train linear layers only')
    parser.add_argument("--clip", type=float, default=2., help='gradient clip, set to -1 if not used')
    parser.add_argument("--wp0", type=float, default=0.005, help='initial lr ratio at the begging of lr warm up')
    parser.add_argument("--wpe", type=float, default=0.01, help='final lr ratio at the end of training')
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--weight_decay_end", type=float, default=0, help='final lr ratio at the end of training')
    parser.add_argument("--resume", type=str, default=False, help='resume')
    parser.add_argument("--ignore_mask", type=bool, default=False, help='ignore_mask')
    parser.add_argument("--val_only", type=bool, default=False, help='validation only')
    parser.add_argument("--c_mask", type=bool, default=False, help='teaching force mask in validation')
    parser.add_argument("--c_img", type=bool, default=False, help='teaching force img in validation')
    parser.add_argument("--cfg", nargs='+', type=float, default=[4, 4, 4], help='cfg guidance scale')
    parser.add_argument("--gibbs", type=int, default=0, help='use gibbs sampling during inference')
    parser.add_argument("--save_val", type=bool, default=False, help='save val images')
    parser.add_argument("--val_cond", type=str, default='depth', help='val condition')
    # vqvae
    parser.add_argument("--vocab_size", type=int, default=4096, nargs='+', help="codebook size")
    parser.add_argument("--z_channels", type=int, default=32, help="latent size of vqvae")
    parser.add_argument("--ch", type=int, default=160, help="channel size of vqvae")
    parser.add_argument("--vqvae_pretrained_path", type=str, default='pretrained/vae_ch160v4096z32.pth', help="vqvae pretrained path")
    parser.add_argument("--var_pretrained_path", type=str, default='pretrained/var_d16.pth', help="var pretrained path")
    # vpq model
    parser.add_argument("--v_patch_nums", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], help="number of patch numbers of each scale")
    parser.add_argument("--v_patch_layers", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], help="index of layers for predicting each scale")
    parser.add_argument("--depth", type=int, default=16, help="depth of vpq model")
    parser.add_argument("--embed_dim", type=int, default=1024, help="embedding dimension of vpq model")
    parser.add_argument("--num_heads", type=int, default=16, help="number of heads of vpq model")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="mlp ratio of vpq model")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="drop rate of vpq model")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, help="attn drop rate of vpq model")
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="drop path rate of vpq model")
    parser.add_argument("--mask_type", type=str, default='interleave_append', help="[interleave_append, replace]")
    parser.add_argument("--uncond", type=bool, default=False, help="uncond gen")
    parser.add_argument("--bidirectional", type=bool, default=False, help="shuffle mask and image order in each stage")
    parser.add_argument("--separate_decoding", type=bool, default=False, help="separate decode mask and image in each stage")
    parser.add_argument("--separator", type=bool, default=False, help="use special tokens as separator")
    parser.add_argument("--type_pos", type=bool, default=False, help="use type pos embed")
    parser.add_argument("--interpos", type=bool, default=False, help="interpolate positional encoding")
    parser.add_argument("--mpos", type=bool, default=False, help="minus positional encoding")
    parser.add_argument("--indep", type=bool, default=False, help="indep separate decoding")
    parser.add_argument("--multi_cond", type=bool, default=False, help="multi-type conditions")
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


def train_epoch(var, vqvae, cond_model, dataloader, optimizer, progress_bar, rank, args):

    var.train()
    if cond_model is not None:
        cond_model.train()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    train_loss = []
    if args.completed_steps != args.epoch * args.num_update_steps_per_epoch:
        print(f'skipping {args.completed_steps - args.epoch * args.num_update_steps_per_epoch} batches')
        print(f'step:{args.completed_steps}, epoch: {args.epoch}, step_per_epoch: {args.num_update_steps_per_epoch}')
    for batch_idx, batch in enumerate(dataloader):
        for _ in range(args.completed_steps - args.epoch * args.num_update_steps_per_epoch):
            continue

        images, masks, conditions, cond_type = batch['image'], batch['mask'], batch['cls'], batch['type']
        images = images.to(device)
        masks = masks.to(device)
        conditions = conditions.to(device)
        cond_type = cond_type.to(device)

        _ = lr_wd_annealing(args.lr_scheduler, optimizer, args.scaled_lr,
                                                             args.weight_decay, args.weight_decay_end,
                                                             args.completed_steps, args.num_warmup_steps,
                                                             args.max_train_steps, wp0=args.wp0, wpe=args.wpe)

        # forward to get input ids
        with torch.no_grad():
            if args.mixed_precision == 'bf16':
                with torch.autocast(device_type='hpu', dtype=torch.bfloat16):
                    mask_labels_list = vqvae.img_to_idxBl(masks, v_patch_nums=args.v_patch_nums)
                    # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
                    mask_input_h_list = vqvae.idxBl_to_h(mask_labels_list)

                    # labels_list: List[(B, 1), (B, 4), (B, 9)]
                    labels_list = vqvae.img_to_idxBl(images, v_patch_nums=args.v_patch_nums)
                    # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
                    input_h_list = vqvae.idxBl_to_h(labels_list)
            else:
                mask_labels_list = vqvae.img_to_idxBl(masks, v_patch_nums=args.v_patch_nums)
                # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
                mask_input_h_list = vqvae.idxBl_to_h(mask_labels_list)

                # labels_list: List[(B, 1), (B, 4), (B, 9)]
                labels_list = vqvae.img_to_idxBl(images, v_patch_nums=args.v_patch_nums)
                # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
                input_h_list = vqvae.idxBl_to_h(labels_list)

        # handle mask
        if args.mask_type == 'replace':
            # Image: r1, r2, r3, Mask: m1, m2, m3
            # New: r1, m2, r3
            # Note that image goes first
            for i in range(len(input_h_list)):
                if i % 2 == 0:
                    labels_list[i] = mask_labels_list[i]
                    input_h_list[i] = mask_input_h_list[i]
            mask_first = False
        elif args.mask_type == 'interleave_append':
            # Image: r1, r2, r3, Mask: m1, m2, m3
            # New: (m1, r1), (m2, r2), (m3, r3)
            # Note that mask goes first unless bidirectional enabled
            if args.bidirectional and random.random() < 0.5:
                labels_list_ = list(chain.from_iterable(zip(labels_list, mask_labels_list)))
                input_h_list_ = list(chain.from_iterable(zip(input_h_list, mask_input_h_list)))
                mask_first = False
            else:
                labels_list_ = list(chain.from_iterable(zip(mask_labels_list, labels_list)))
                input_h_list_ = list(chain.from_iterable(zip(mask_input_h_list, input_h_list)))
                mask_first = True
            labels_list, input_h_list = labels_list_, input_h_list_
        else:
            raise NotImplementedError

        x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1)

        # forwad through model
        if args.mixed_precision == 'bf16':
            with torch.autocast(device_type='hpu', dtype=torch.bfloat16):
                logits = var(conditions, x_BLCv_wo_first_l, mask_first=mask_first, cond_type=cond_type)  # BLC, C=vocab size
        else:
            logits = var(conditions, x_BLCv_wo_first_l, mask_first=mask_first, cond_type=cond_type)  # BLC, C=vocab size
        logits = logits.view(-1, logits.size(-1))

        if args.separator:
            mapping = [i for i in range(18)] if mask_first else [i + 1 if i % 2 == 0 else i - 1 for i in range(18)]
            B = labels_list[0].shape[0]
            label1, label2 = labels_list[0], labels_list[1]
            new_labels_list = [label1, label2]
            for i, label in enumerate(labels_list[2:]):
                special_label = label1.new_ones(B, 1) * (mapping[i] + args.vocab_size)
                if i % 2 == 0:
                    new_labels_list.extend([label, special_label])
                else:
                    new_labels_list.extend([label, special_label])
            labels_list = new_labels_list


        labels = torch.cat(labels_list, dim=1)
        labels = labels.view(-1)
        print(logits)
        loss = loss_fn(logits, labels)

        if args.ignore_mask:
            ignore_mask = batch['ignore_mask'] if mask_first else batch['ignore_mask_']
            ignore_mask.to(device)
            ignore_mask = ignore_mask.view(-1)
            loss = (loss * ignore_mask.float()).mean() / (ignore_mask.mean() + 1e-6)
        else:
            loss = loss.mean()

        loss.backward()
        htcore.mark_step()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(var.parameters(), args.clip)

        optimizer.step()
        htcore.mark_step()
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()
            progress_bar.set_description(f"train/loss: {loss.item()}")
        args.completed_steps += 1
        progress_bar.update(1)

        train_loss.append(loss.item())

        if rank == 0:
            # Log metrics
            if args.completed_steps % args.log_interval == 0:
                train_loss_mean = torch.tensor(sum(train_loss) / len(train_loss))  #.to(device)
                # dist.all_reduce(train_loss_mean, op=dist.ReduceOp.SUM)
                wandb.log(
                    {
                        "train/loss": train_loss_mean.item(),
                        "step": args.completed_steps,
                        "epoch": args.epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                        "weight_decay": optimizer.param_groups[0]["weight_decay"],
                    },
                    step=args.completed_steps)
                inference(var, vqvae, cond_model, np.random.choice(args.num_classes, 4).tolist(), rank=rank,
                          guidance_scale=4.0, top_k=900, top_p=0.95, seed=42)

            # Save model
            if isinstance(args.save_interval, int):
                if args.completed_steps % args.save_interval == 0:
                    save_dir = os.path.join(args.project_dir, f"step_{args.completed_steps}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_checkpoint(var, optimizer, args, latest=True)


@torch.no_grad()
def inference(var, vqvae, cond_model, conditions, rank=0, guidance_scale=4.0, top_k=900, top_p=0.95, seed=42):
    var.eval()
    if cond_model:
        cond_model.eval()
    # conditions = [474, 474, 474, 474]
    images = var.module.autoregressive_infer_cfg(B=len(conditions), label_B=torch.tensor(conditions, device=device),
                                          cfg=guidance_scale, top_k=top_k, top_p=top_p, g_seed=seed)
    image = make_grid(images, nrow=len(conditions), padding=0, pad_value=1.0)
    image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
    image = Image.fromarray(image.astype(np.uint8))

    wandb.log({f"images": [wandb.Image(image, caption=f"{conditions}")]})

    var.train()
    # if cond_model:
    #     cond_model.train()

def pix_cond_inference(images, masks, conditions, cond_type, device, B, var, vqvae, c_mask, c_img,
                       guidance_scale, top_k, top_p, seed, args):
    types = {'mask': 0, 'canny': 1, 'depth': 2, 'normal': 3, 'none': 4}
    images = images.to(device)
    masks = masks.to(device)
    if isinstance(conditions, int):
        conditions = torch.tensor([conditions for _ in range(B)]).to(device)
    else:
        conditions = conditions.to(device)  # cls
    if isinstance(cond_type, str):
        cond_type = torch.tensor([types[cond_type] for _ in range(B)], device=var.device)
    else:
        cond_type = cond_type.to(device)

    with torch.no_grad():
        if c_mask:
            c_mask = vqvae.img_to_idxBl(masks, v_patch_nums=args.v_patch_nums)
        elif c_img:
            c_img = vqvae.img_to_idxBl(images, v_patch_nums=args.v_patch_nums)
        else:
            c_mask, c_img = None, None

        images = var.module.conditional_infer_cfg(B=B, label_B=conditions, cfg=guidance_scale, top_k=top_k,
                                                  top_p=top_p, g_seed=seed, c_mask=c_mask, c_img=c_img, cond_type=cond_type)
        htcore.mark_step()
    return images

def cls_cond_inference(cls, device, B, var, index, cond_type, guidance_scale, top_k, top_p, seed):
    types = {'mask': 0, 'canny': 1, 'depth': 2, 'normal': 3, 'none': 4}
    conditions = torch.tensor([cls for _ in range(B)], device=device).long()
    cond_type = torch.tensor([types[cond_type] for _ in range(B)], device=var.device).long()
    with torch.no_grad():
        images = var.module.autoregressive_infer_cfg(B=B, label_B=conditions,
                                                     cond_type=cond_type, cfg=guidance_scale[0],
                                                     top_k=top_k, top_p=top_p, g_seed=seed)
        htcore.mark_step()
    return images

def validate(var, vqvae, cond_model, dataloader, args, guidance_scale=(6, 6, 6), top_k=900, top_p=0.96, seed=42,
             c_mask=None, c_img=None, rank=0, gibbs=0, save_val=True):
    var.eval()
    if cond_model:
        cond_model.eval()
    if c_mask or c_img:
        pbar = tqdm(range(math.ceil(len(dataloader))), disable=not rank == 0)
        save_path = os.path.join(args.project_dir, f'cfg_{guidance_scale[0]}_{guidance_scale[1]}_{guidance_scale[2]}_{args.val_cond}',
                                 f'{rank}')
        os.makedirs(save_path, exist_ok=True)
        for batch_idx, batch in enumerate(dataloader):
            images, masks, conditions, cond_type = batch['image'], batch['mask'], batch['cls'], batch['type']
            B = masks.shape[0]
            images = pix_cond_inference(images, masks, conditions, cond_type, device, B, var, vqvae, c_mask, c_img,
                       guidance_scale, top_k, top_p, seed, args)
            if save_val:
                images = images.permute(0, 2, 3, 1).mul_(255).cpu().numpy().astype(np.uint8)
                for b in range(B):
                    image = Image.fromarray(images[b, 256:])
                    image.save(os.path.join(save_path, f'{batch_idx * B + b}.png'))
            else:
                image_ = make_grid(images, nrow=B, padding=0, pad_value=1.0)
                image_ = image_.permute(1, 2, 0).mul_(255).cpu().numpy()
                image_ = Image.fromarray(image_.astype(np.uint8))
                wandb.log({f"images": [wandb.Image(image_, caption=f"{conditions}_{guidance_scale}")]})

            pbar.update(1)
    else:
        slices = 1000 // args.gpus
        classes = [i for i in range(slices * rank, slices * (rank + 1))] if rank != args.gpus - 1 \
            else [i for i in range(slices * rank, 1000)]
        pbar = tqdm(range(len(classes)), disable=not rank == 0)
        for cls in classes:
            os.makedirs(os.path.join(args.project_dir, f'cfg_{guidance_scale[0]}', f'{cls}'), exist_ok=True)
            assert 50 > args.batch_size
            for i in range(50 // args.batch_size + 1):
                B = args.batch_size if i != 50 // args.batch_size else 50 - i * args.batch_size
                if B == 0: continue
                cond_type = 'depth'
                seed = seed + i * (cls + 1)
                images = cls_cond_inference(cls, device, B, var, i, cond_type, guidance_scale, top_k, top_p, seed)
                # image = make_grid(images, nrow=B, padding=0, pad_value=1.0)
                if gibbs != 0:
                    for g_step in range(gibbs):
                        # start from mask teaching force
                        masks, images = images[:, :, :256, :], images[:, :, 256:, :]
                        masks, images = (masks - 0.5) / 0.5, (images - 0.5) / 0.5
                        c_mask = True
                        images = pix_cond_inference(images, masks, cls, cond_type, device, B, var, vqvae, c_mask,
                                                    c_img, guidance_scale, top_k, top_p, seed, args)

                        masks, images = images[:, :, :256, :], images[:, :, 256:, :]
                        masks, images = (masks - 0.5) / 0.5, (images - 0.5) / 0.5
                        c_img = True
                        images = pix_cond_inference(images, masks, cls, cond_type, device, B, var, vqvae, c_mask,
                                                    c_img, guidance_scale, top_k, top_p, seed, args)

                if save_val:
                    images = images.permute(0, 2, 3, 1).mul_(255).cpu().numpy().astype(np.uint8)
                    for b in range(B):
                        image = Image.fromarray(images[b, 256])
                        image.save(os.path.join(args.project_dir, f'cfg_{guidance_scale[0]}', f'{cls}',
                                                f'{i * args.batch_size + b}.png'))
                else:
                    image_ = make_grid(images, nrow=B, padding=0, pad_value=1.0)
                    image_ = image_.permute(1, 2, 0).mul_(255).cpu().numpy()
                    image_ = Image.fromarray(image_.astype(np.uint8))
                    wandb.log({f"images": [wandb.Image(image_, caption=f"{cls}_{guidance_scale}")]})
            pbar.update(1)

    var.train()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'
    # initialize the process group
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, args, save_dir='', latest=False):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epoch,
        'step': args.completed_steps
    }
    step = 'latest' if latest else args.completed_steps
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_step_{step}.pth'))

def resume(var, optimizer, args):
    state_dict = torch.load(args.resume, map_location=torch.device('cpu'))
    if 'model_state_dict' in state_dict.keys():
        var_state_dict = state_dict['model_state_dict']

        var.load_state_dict(var_state_dict, strict=True)

    if 'optimizer_state_dict' in state_dict.keys():
        opt_state_dict = state_dict['optimizer_state_dict']
        optimizer.load_state_dict(opt_state_dict)

    args.completed_steps = state_dict['step']
    args.starting_epoch = state_dict['epoch']

    if 'latest' not in args.resume:
        args.starting_epoch += 1

    print(f'Resume from step: {args.completed_steps}, epoch: {args.starting_epoch}')

def prepare_lora(var):
    print('Warning: The weights in attn.mat_kqv are currently not supported.')
    # TODO: attn.mat_kqv
    lora_params = []
    for name, _ in var.named_modules():
        if ('attn.' in name and 'attn.proj_drop' not in name and 'attn.mat_qkv' not in name) or \
                'ffn.fc' in name or \
                'ada_lin.1' in name or \
                'head_nm.ada_lin.1' in name:
            lora_params.append(name)
    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=lora_params,
        lora_dropout=0.05,
        bias="none",
    )
    # add LoRA adaptor
    # var = prepare_model_for_kbit_training(var)
    var = get_peft_model(var, lora_config)
    var.print_trainable_parameters()

def load_var_weight(var, args):
    var_state_dict = torch.load(args.var_pretrained_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in var_state_dict.keys():
        var_state_dict = var_state_dict['model_state_dict']
    new_dict = OrderedDict()
    for k, v in var_state_dict.items():
        new_dict[k.replace('module.', '')] = v
    var_state_dict = new_dict
    init_std = math.sqrt(1 / args.embed_dim / 3)
    print('load model')
    if args.mask_type == 'interleave_append':
        print('handle pos and attn bias')
        for key in ['lvl_1L', 'pos_start', 'attn_bias_for_masking']:
            del var_state_dict[key]  # will be handled in the init
        print('handle pos_1LC')
        for key in ['pos_1LC', ]:
            pos_1LC_ = var_state_dict[key]
            if args.interpos:
                # pos_1LC = var_state_dict[key].permute(0, 2, 1)  # 1, C, L
                # var_state_dict[key] = torch.nn.functional.interpolate(pos_1LC, size=1378).permute(0, 2, 1)
                # var_state_dict[key] = torch.concat([var_state_dict[key], var_state_dict[key], var_state_dict[key][:,:18]], dim=1)
                pos_1LC = []
                L = 0
                for i, pn in enumerate(args.v_patch_nums):
                    pe = torch.empty((pn * pn) * 2, args.embed_dim)
                    nn.init.trunc_normal_(pe, mean=0, std=init_std)
                    pe[:pn * pn] = pos_1LC_[:, L:L + pn * pn]
                    pe[pn * pn:pn * pn * 2] = pos_1LC_[:, L:L + pn * pn]
                    pos_1LC.append(pe)
                    L += pn * pn
                pos_1LC = torch.cat(pos_1LC, dim=0).unsqueeze(0)  # 1, L, C
                var_state_dict[key] = pos_1LC
            else:
                if args.separator:
                    pos_1LC = []
                    L = 0
                    for i, pn in enumerate(args.v_patch_nums):
                        num_sp_tokens = 1 if i != 0 else 0
                        pe = torch.empty((pn * pn + num_sp_tokens) * 2, args.embed_dim)
                        nn.init.trunc_normal_(pe, mean=0, std=init_std)
                        pe[:pn * pn] = pos_1LC_[:, L:L + pn * pn]
                        pe[pn * pn + num_sp_tokens:pn * pn * 2 + num_sp_tokens] = pos_1LC_[:,
                                                                                  L:L + pn * pn] * -1 if args.mpos else 1
                        pos_1LC.append(pe)
                        L += pn * pn
                    pos_1LC = torch.cat(pos_1LC, dim=0).unsqueeze(0)  # 1, L, C
                    var_state_dict[key] = pos_1LC
                else:

                    var_state_dict[key] = torch.concat([var_state_dict[key], var_state_dict[key]], dim=1)

        if args.separator:
            weight = torch.empty(args.vocab_size + (len(args.v_patch_nums) - 1) * 2, args.embed_dim)
            bias = torch.empty(args.vocab_size + (len(args.v_patch_nums) - 1) * 2)
            nn.init.trunc_normal_(weight, mean=0, std=init_std)
            nn.init.trunc_normal_(bias, mean=0, std=init_std)
            weight = weight.mul_(0.02)
            bias = bias.mul_(0.0)
            weight[:args.vocab_size] = var_state_dict['head.weight']
            bias[:args.vocab_size] = var_state_dict['head.bias']
            var_state_dict['head.weight'] = weight
            var_state_dict['head.bias'] = bias
    var.load_state_dict(var_state_dict, strict=False)

def process(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    seed_everything(rank)

    if rank == 0:
        if args.debug:
            wandb.init(project="Debug")
        else:
            wandb.init(project="ControlVAR")

    # Setup accelerator:
    if args.run_name is None:
        model_name = f'vqvae_ch{args.ch}v{args.vocab_size}z{args.z_channels}_maskvar_d{args.depth}e{args.embed_dim}h{args.num_heads}_{args.dataset_name}_ep{args.num_epochs}_bs{args.batch_size}_clip{args.clip}'
    else:
        model_name = args.run_name

    args.model_name = model_name
    args.embed_dim = args.depth * 64
    timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.project_dir = f"experiments/{args.output_dir}"  # Create an experiment folder
    os.makedirs(args.project_dir, exist_ok=True)
    save_interval = args.save_interval

    if save_interval is not None and save_interval.isdigit():
        save_interval = int(save_interval)
        args.save_interval = save_interval

    # create dataset
    print(f"Creating dataset {args.dataset_name}")
    dataset = create_dataset(args.dataset_name, args)
    val_dataset = create_dataset(args.dataset_name, args, split='val')
    # create dataloader
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    # Calculate total batch size
    total_batch_size = args.batch_size * args.gpus * args.gradient_accumulation_steps
    args.total_batch_size = total_batch_size

    # Create VQVAE Model
    print("Creating VQVAE model")
    vqvae = VQVAE(vocab_size=args.vocab_size, z_channels=args.z_channels, ch=args.ch, test_mode=True,
                  share_quant_resi=4, v_patch_nums=args.v_patch_nums,).to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    if args.vqvae_pretrained_path is not None:
        vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path, map_location=torch.device('cpu')))

    # Create VPA Model
    print("Creating VAR model")

    var = build_control_var(vae=vqvae, depth=args.depth, patch_nums=args.v_patch_nums, mask_type=args.mask_type,
                         cond_drop_rate=1.1 if args.uncond else 0.1, bidirectional=args.bidirectional,
                         separate_decoding=args.separate_decoding, separator=args.separator, type_pos=args.type_pos,
                         indep=args.indep, multi_cond=args.multi_cond)

    if args.var_pretrained_path is not None and not args.resume:
        print('Loading varmodel')
        load_var_weight(var, args)

    if args.lora:
        prepare_lora()

    var = DDP(var.to(device), find_unused_parameters=False)
    var.train()


    print('Filtering parameters')
    names, paras, para_groups = filter_params(var, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })

    # Create Condition Model
    print("Creating conditional model")
    if args.condition_model is None:
        cond_model = None
    elif args.condition_model == 'class_embedder':
        from models.class_embedder import ClassEmbedder
        cond_model = ClassEmbedder(num_classes=args.num_classes, embed_dim=args.embed_dim, cond_drop_rate=args.cond_drop_rate)
    else:
        raise NotImplementedError(f"Condition model {args.condition_model} is not implemented")

    # Create Optimizer
    print("Creating optimizer")
    # TODO: support faster optimizer

    args.scaled_lr = args.learning_rate * total_batch_size / 512
    optimizer = torch.optim.AdamW(para_groups, lr=args.scaled_lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
    # Compute max_train_steps
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # Create Learning Rate Scheduler
    args.num_warmup_steps = int(args.wp0 * args.max_train_steps) if args.lr_warmup_steps < 1.0 else int(args.lr_warmup_steps)
    args.num_update_steps_per_epoch = num_update_steps_per_epoch
    # Start training
    if rank == 0:
        print("***** Training arguments *****")
        print(args)
        print("***** Running training *****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Num Epochs = {args.num_epochs}")
        print(f"  Instantaneous batch size per device = {args.batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        print(f"  Total optimization steps = {args.max_train_steps}")
        print(f"  Scaled learning rate = {args.learning_rate * total_batch_size / 512}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not rank == 0)
    args.completed_steps = 0
    args.starting_epoch = 0

    if args.resume:
        resume(var, optimizer, args)
        progress_bar.update(args.completed_steps)
        print(f'resume from step {args.completed_steps}')

    if rank == 0 and not args.val_only:
        print('start eval')
        inference(var, vqvae, cond_model, np.random.choice(args.num_classes, 4).tolist(),
                  guidance_scale=4.0, top_k=900, top_p=0.96, seed=42)
        print('end eval')

    if not args.val_only:
        # Training
        for epoch in range(args.starting_epoch, args.num_epochs):

            args.epoch = epoch
            if rank == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}")
            train_epoch(var, vqvae, cond_model, dataloader, optimizer, progress_bar, rank, args)

            if epoch % args.val_interval == 0 and rank == 0:
                inference(var, vqvae, cond_model, np.random.choice(args.num_classes, 4).tolist(),
                          guidance_scale=4.0, top_k=900, top_p=0.96, seed=42)

            if args.save_interval == 'epoch' and rank == 0:
                save_checkpoint(var, optimizer, args, args.project_dir)
    else:
        assert not (args.c_img and args.c_mask)  # only give one condition
        validate(var, vqvae, cond_model, val_dataloader, args, c_mask=args.c_mask,
                 c_img=args.c_img, rank=rank, guidance_scale=args.cfg, gibbs=args.gibbs, save_val=args.save_val)
    # end training
    cleanup()


def run(process, world_size, args):
    mp.set_start_method('spawn')
    mp.spawn(process,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    args = parse_args()
    run(process, args.gpus, args)
    # args.completed_steps = 0
    # args.starting_epoch = 0
    # while args.starting_epoch < 30:
    #     try:
    #         run(process, args.gpus, args)
    #     except:
    #         args.resume = f'experiments/{args.output_dir}/checkpoint_step_latest.pth'
