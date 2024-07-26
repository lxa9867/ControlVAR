import os
import argparse
import math
import numpy as np
from itertools import chain
from time import time
from datetime import datetime
from tqdm.auto import tqdm
# import wandb
from PIL import Image 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from transformers import get_scheduler

from datasets import create_dataset
from models.vqvae_mask import VQVAE
from losses.vqperceptual import VQLPIPSWithDiscriminator
from ruamel.yaml import YAML

device = torch.device('cuda')

def parse_args():
    parser = argparse.ArgumentParser()
    
    # config file
    parser.add_argument("--config", type=str, default=None, help="config file used to specify parameters")
    
    # data
    parser.add_argument("--data", type=str, default=None, help="data")
    parser.add_argument("--data_dir", type=str, default='/mnt/data/ImageNetS919', help="data folder")
    parser.add_argument("--dataset_name", type=str, default="imagenetS", help="dataset name")
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--batch_size", type=int, default=2, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    
    # training
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=4.5e-6, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--lr_scheduler", type=str, default='linear', help='lr scheduler')
    parser.add_argument("--lr_warmup_steps", type=float, default=0., help="warmup steps")
    parser.add_argument("--log_interval", type=int, default=5, help='log interval for steps')
    parser.add_argument("--val_interval", type=int, default=1, help='validation interval for epochs')
    parser.add_argument("--save_interval", type=str, default='5000', help='save interval')
    parser.add_argument("--mixed_precision", type=str, default='no', help='mixed precision', choices=['no', 'fp16', 'bf16', 'fp8'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')
    
    # vqvae
    parser.add_argument("--vocab_size", type=int, default=4096, nargs='+', help="codebook size")
    parser.add_argument("--z_channels", type=int, default=32, help="latent size of vqvae")
    parser.add_argument("--ch", type=int, default=160, help="channel size of vqvae")
    parser.add_argument("--vqvae_pretrained_path", type=str, default=None, help="vqvae pretrained path")
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

    # resume args
    parser.add_argument("--resume", type=str, default=None, help="Resume function")
    
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


def train_epoch(vqvae, loss_fn, dataloader, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, progress_bar, rank, args):

    vqvae.train()
    loss_fn.train()
    for batch_idx, batch in enumerate(dataloader):
        images, masks, conditions = batch['image'], batch['mask'], batch['cls']
        images = images.to(device)
        masks = masks.to(device)

        # forwad through model
        recon_images, recon_mask, usages, mvq_loss, vq_loss = vqvae(images, masks)  # BLC, C=vocab size

        aeloss, log_dict_ae = loss_fn(mvq_loss, vq_loss, images, masks, recon_images, recon_mask, 0, 
        args.completed_steps, last_layer=vqvae.module.get_last_layer(), split="train")

        aeloss.backward()
        optimizer_G.step()
        optimizer_G.zero_grad()
        lr_scheduler_G.step()

        discloss, log_dict_disc = loss_fn(mvq_loss, vq_loss, images, masks, recon_images, recon_mask, 1, 
        args.completed_steps, last_layer=vqvae.module.get_last_layer(), split="train")

        discloss.backward()
        optimizer_D.step()
        optimizer_D.zero_grad()
        lr_scheduler_D.step()

        progress_bar.update(1)
        args.completed_steps += 1

        if rank == 0:
            # Log metrics
            if args.completed_steps % args.log_interval == 0:
                wandb.log({f"autoencoder/{key}": value for key, value in log_dict_ae.items()}, step=args.completed_steps)
                image = torch.cat([images, recon_images, masks, recon_mask], dim=0)
                image = torch.clamp(image, min=-1, max=1)
                image = make_grid((image + 1) / 2, nrow=images.shape[0], padding=0, pad_value=1.0)
                image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
                image = Image.fromarray(image.astype(np.uint8))
                wandb.log({f"images": [wandb.Image(image)]}, step=args.completed_steps)


            # Save model
            if isinstance(args.save_interval, int):
                if args.completed_steps % args.save_interval == 0:
                    save_dir = os.path.join(args.project_dir, f"step_{args.completed_steps}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_checkpoint(vqvae, loss_fn, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, args, -1, args.completed_steps,)
            #
            # TODO remove
            # if args.completed_steps % 100 == 0:
            #     inference(var, vqvae, cond_model, np.random.choice(args.num_classes, 4).tolist(), rank=rank,
            #               guidance_scale=4.0, top_k=900, top_p=0.95, seed=42)


@torch.no_grad()
def inference(vqvae, images, rank=0, guidance_scale=4.0, seed=42):
    vqvae.eval()
    # conditions = [474, 474, 474, 474]
    recon_images = vqvae(images)
    result = make_grid(torch.cat((images, recon_images), dim=0), nrow=images.shape[0] * 2, padding=0, pad_value=1.0)
    result = result.permute(1, 2, 0).mul_(255).cpu().numpy()
    result = result.fromarray(result.astype(np.uint8))

    wandb.log({f"images": [wandb.Image(result, caption="images and reconstruction")]})

    vqvae.train()


def validate():
    pass


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12347'
    # initialize the process group
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D, args, epoch=None, step=None, ):
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
        'epoch': epoch,
        'step': step
    }
    torch.save(checkpoint, f'{args.output_dir}/checkpoint_step_{step:08d}.pth')

def process(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    if rank == 0:
        wandb.init(project="MaskVAE")

    # Setup accelerator:
    if args.run_name is None:
        model_name = f'vqvae_ch{args.ch}v{args.vocab_size}z{args.z_channels}_maskvar_d{args.depth}e{args.embed_dim}h{args.num_heads}_{args.dataset_name}_ep{args.num_epochs}_bs{args.batch_size}'
    else:
        model_name = args.run_name

    args.model_name = model_name
    timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.project_dir = f"{args.output_dir}/{timestamp}-{model_name}"  # Create an experiment folder
    os.makedirs(args.project_dir, exist_ok=True)
    save_interval = args.save_interval

    if save_interval is not None and save_interval.isdigit():
        save_interval = int(save_interval)
        args.save_interval = save_interval

    # create dataset
    print("Creating dataset")
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
                  share_quant_resi=4, v_patch_nums=args.v_patch_nums).to(device)
    vqvae.train()
    for p in vqvae.parameters():
        p.requires_grad_(True)
    if args.vqvae_pretrained_path is not None:
        vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path))
    vqvae = DDP(vqvae)

    # Create Discriminator
    print("Creating discriminator")
    loss_fn = VQLPIPSWithDiscriminator(disc_conditional=False, 
    disc_in_channels=3,disc_start=500000,disc_weight=0.8,codebook_weight=1.0).to(device)
    loss_fn = DDP(loss_fn)

    # Create Optimizer
    print("Creating optimizer")
    # TODO: support faster optimizer
    trainable_params_G = list(vqvae.parameters())
    trainable_params_D = list(loss_fn.parameters())

    optimizer_G = torch.optim.Adam(trainable_params_G, lr=args.learning_rate, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(trainable_params_D, lr=args.learning_rate, betas=(0.5, 0.9))
    # Compute max_train_steps
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # Create Learning Rate Scheduler
    print("Creating learning rate scheduler")
    num_warmup_steps = int(args.lr_warmup_steps * args.max_train_steps) if args.lr_warmup_steps < 1.0 else int(args.lr_warmup_steps)
    lr_scheduler_G = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer_G,
        num_warmup_steps=num_warmup_steps * args.gpus,
        num_training_steps=args.max_train_steps
    )
    lr_scheduler_D = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer_D,
        num_warmup_steps=num_warmup_steps * args.gpus,
        num_training_steps=args.max_train_steps
    )
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
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not rank == 0)
    args.completed_steps = 0
    args.starting_epoch = 0

    # TODO: add resume function

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        vqvae.load_state_dict(checkpoint['generator_state_dict'])
        loss_fn.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        lr_scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        lr_scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        args.completed_steps = checkpoint['step']
        args.starting_epoch = checkpoint['epoch']
        progress_bar.update(args.completed_steps)

        
    # if rank == 0:
    #     print('start eval')
    #     images = dataloader[0]
    #     inference(vqvae, images, seed=42)
    #     print('end eval')
    # Training
    for epoch in range(args.starting_epoch, args.num_epochs):

        args.epoch = epoch
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}")

        # train epoch
        train_epoch(vqvae, loss_fn, dataloader, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, progress_bar, rank, args)

        # if epoch % args.val_interval == 0 and rank == 0:
        #     inference(vqvae, images, seed=42)

        if args.save_interval  == 'epoch' and rank == 0:
            save_dir = os.path.join(args.project_dir, f"epoch_{args.epoch}")
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(vqvae, loss_fn, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, args, epoch, args.completed_steps)
    
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
