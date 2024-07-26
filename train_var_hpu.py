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
from models import VQVAE, VisualProgressAutoreg, VAR, build_var, MaskVAR, build_mask_var
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
    parser.add_argument("--save_interval", type=str, default='10000', help='save interval')
    parser.add_argument("--mixed_precision", type=str, default='bf16', help='mixed precision', choices=['no', 'fp16', 'bf16', 'fp8'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')
    parser.add_argument("--lora", type=bool, default=False, help='use lora to train linear layers only')
    parser.add_argument("--clip", type=float, default=2., help='gradient clip, set to -1 if not used')
    parser.add_argument("--wp0", type=float, default=0.005, help='initial lr ratio at the begging of lr warm up')
    parser.add_argument("--wpe", type=float, default=0.01, help='final lr ratio at the end of training')
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--weight_decay_end", type=float, default=0, help='final lr ratio at the end of training')
    parser.add_argument("--resume", type=bool, default=False, help='resume')
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
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loss = []
    for batch_idx, batch in enumerate(dataloader):
        images, masks, conditions = batch['image'], batch['mask'], batch['cls']
        images = images.to(device)
        conditions = conditions.to(device)

        _ = lr_wd_annealing(args.lr_scheduler, optimizer, args.scaled_lr,
                                                             args.weight_decay, args.weight_decay_end,
                                                             args.completed_steps, args.num_warmup_steps,
                                                             args.max_train_steps, wp0=args.wp0, wpe=args.wpe)

        # forward to get input ids
        with torch.no_grad():
            if args.mixed_precision == 'bf16':
                with torch.autocast(device_type='hpu', dtype=torch.bfloat16):
                    # labels_list: List[(B, 1), (B, 4), (B, 9)]
                    labels_list = vqvae.img_to_idxBl(images, v_patch_nums=args.v_patch_nums)
                    # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
                    input_h_list = vqvae.idxBl_to_h(labels_list)
            else:
                # labels_list: List[(B, 1), (B, 4), (B, 9)]
                labels_list = vqvae.img_to_idxBl(images, v_patch_nums=args.v_patch_nums)
                # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
                input_h_list = vqvae.idxBl_to_h(labels_list)


        x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1)

        # forwad through model
        if args.mixed_precision == 'bf16':
            with torch.autocast(device_type='hpu', dtype=torch.bfloat16):
                logits = var(conditions, x_BLCv_wo_first_l)  # BLC, C=vocab size
        else:
            logits = var(conditions, x_BLCv_wo_first_l)  # BLC, C=vocab size
        logits = logits.view(-1, logits.size(-1))

        labels = torch.cat(labels_list, dim=1)
        labels = labels.view(-1)
        loss = loss_fn(logits, labels)
        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(var.parameters(), args.clip)

        htcore.mark_step()
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
                    save_checkpoint(var, optimizer, args, -1, args.completed_steps,)


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


def validate():
    pass


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    # initialize the process group
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, args, epoch=None, step=None, save_dir=''):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_step_{step}.pth'))

def process(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    seed_everything(rank)

    if rank == 0:
        wandb.login(key='db60d37b5f7529afeab349ab23441b7888cceee6')
        if args.debug:
            wandb.init(project="Debug")
        else:
            wandb.init(project="MaskVAR")

    # Setup accelerator:
    if args.run_name is None:
        model_name = f'vqvae_ch{args.ch}v{args.vocab_size}z{args.z_channels}_maskvar_d{args.depth}e{args.embed_dim}h{args.num_heads}_{args.dataset_name}_ep{args.num_epochs}_bs{args.batch_size}_clip{args.clip}'
    else:
        model_name = args.run_name

    args.model_name = model_name
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
    # create dataloader
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
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

    var = build_var(vae=vqvae, depth=args.depth, patch_nums=args.v_patch_nums)

    if args.var_pretrained_path is not None:
        var_state_dict = torch.load(args.var_pretrained_path, map_location=torch.device('cpu'))
        var.load_state_dict(var_state_dict, strict=True)

    if args.lora:
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

    var = DDP(var.to(device), find_unused_parameters=False)
    var.train()

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

    # TODO: add resume function
    if rank == 0:
        print('start eval')
        inference(var, vqvae, cond_model, np.random.choice(args.num_classes, 4).tolist(),
                  guidance_scale=4.0, top_k=900, top_p=0.95, seed=42)
        print('end eval')

    # Training
    for epoch in range(args.starting_epoch, args.num_epochs):

        args.epoch = epoch
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}")
        train_epoch(var, vqvae, cond_model, dataloader, optimizer, progress_bar, rank, args)

        if epoch % args.val_interval == 0 and rank == 0:
            inference(var, vqvae, cond_model, np.random.choice(args.num_classes, 4).tolist(),
                      guidance_scale=4.0, top_k=900, top_p=0.95, seed=42)

        if args.save_interval == 'epoch' and rank == 0:
            save_checkpoint(var, optimizer, args, epoch, args.completed_steps, args.project_dir)

    # end training
    cleanup()


def run(process, world_size, args):
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(process,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    args = parse_args()
    run(process, args.gpus, args)
