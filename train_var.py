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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from datasets import create_dataset
from models import VQVAE, VisualProgressAutoreg, VAR, build_var
from utils.wandb import CustomWandbTracker
from ruamel.yaml import YAML

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument("--config", type=str, default='configs/train_var_ImageNet_local.yaml',
                        help="config file used to specify parameters")

    # data
    parser.add_argument("--data", type=str, default=None, help="data")
    parser.add_argument("--data_dir", type=str, default='/mnt/data/ImageNet2012', help="data folder")
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
    parser.add_argument("--var_pretrained_path", type=str, default='pretrained/var_d16.pth', help="var pretrained path")
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


def train_epoch(accelerator, var, vqvae, cond_model, dataloader, optimizer, lr_scheduler, progress_bar, args):
    var.train()
    if cond_model is not None:
        cond_model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(dataloader):

        with accelerator.accumulate(var):
            images, conditions = batch
            # forward to get input ids
            with torch.no_grad():
                # labels_list: List[(B, 1), (B, 4), (B, 9)]
                labels_list = vqvae.img_to_idxBl(images, v_patch_nums=args.v_patch_nums)
                # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
                input_h_list = vqvae.idxBl_to_h(labels_list)
                x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1)

            # forwad through model
            logits = var(conditions, x_BLCv_wo_first_l)  # BLC, C=vocab size
            logits = logits.view(-1, logits.size(-1))
            labels = torch.cat(labels_list, dim=1)
            labels = labels.view(-1)

            loss = loss_fn(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            args.completed_steps += 1

        # Log metrics
        if args.completed_steps % args.log_interval == 0:
            accelerator.log(
                {
                    "train/loss": loss.item(),
                    "step": args.completed_steps,
                    "epoch": args.epoch,
                    "lr": optimizer.param_groups[0]["lr"]
                },
                step=args.completed_steps)

        # Save model
        if isinstance(args.save_interval, int):
            if args.completed_steps % args.save_interval == 0:
                save_dir = os.path.join(args.project_dir, f"step_{args.completed_steps}")
                os.makedirs(save_dir, exist_ok=True)
                accelerator.save_state(save_dir)

        # TODO remove
        if args.completed_steps % 100 == 0:
            inference(accelerator, var, vqvae, cond_model, np.random.choice(args.num_classes, 4).tolist(),
                      num_samples=1, guidance_scale=4.0, top_k=900, top_p=0.95, seed=42)


@torch.no_grad()
def inference(accelerator, var, vqvae, cond_model, conditions,
              num_samples=1, guidance_scale=4.0, top_k=900, top_p=0.95, seed=42):
    var.eval()
    vqvae.eval()
    cond_model.eval()
    images = var.autoregressive_infer_cfg(B=len(conditions),
                                          label_B=torch.tensor(conditions, device=torch.device('cuda')),
                                          cfg=4, top_k=top_k, top_p=top_p, g_seed=seed)
    image = make_grid(images, nrow=len(conditions), padding=0, pad_value=1.0)
    image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
    image = Image.fromarray(image.astype(np.uint8))

    accelerator.log({"images": [wandb.Image(image, caption=f"{conditions}")]})
    var.train()
    vqvae.train()
    cond_model.train()


def validate():
    pass


def main():
    args = parse_args()

    # seed
    set_seed(args.seed)

    # Setup accelerator:
    if args.run_name is None:
        model_name = f'vqvae_ch{args.ch}v{args.vocab_size}z{args.z_channels}_vpa_d{args.depth}e{args.embed_dim}h{args.num_heads}_{args.dataset_name}_ep{args.num_epochs}_bs{args.batch_size}'
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

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=CustomWandbTracker(model_name),
        project_dir=args.project_dir)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # create dataset
    logger.info("Creating dataset")
    dataset = create_dataset(args.dataset_name, args)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
    # Calculate total batch size
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    args.total_batch_size = total_batch_size

    # Create VQVAE Model
    logger.info("Creating VQVAE model")
    vqvae = VQVAE(vocab_size=args.vocab_size, z_channels=args.z_channels, ch=args.ch, test_mode=True,
                  share_quant_resi=4, v_patch_nums=args.v_patch_nums)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    if args.vqvae_pretrained_path is not None:
        vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path))

    # Create VPA Model
    logger.info("Creating VAR model")
    # var = VAR(vocab_size=args.vocab_size, depth=args.depth, embed_dim=args.embed_dim, num_heads=args.num_heads,
    #                             mlp_ratio=args.mlp_ratio, drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate,
    #                             v_patch_nums=args.v_patch_nums, v_patch_layers=args.v_patch_layers)
    var = build_var(vae=vqvae, depth=args.depth, patch_nums=args.v_patch_nums)
    if args.var_pretrained_path is not None:
        var.load_state_dict(torch.load(args.var_pretrained_path))
    var.train()

    # Create Condition Model
    logger.info("Creating conditional model")
    if args.condition_model is None:
        cond_model = None
    elif args.condition_model == 'class_embedder':
        from models.class_embedder import ClassEmbedder
        cond_model = ClassEmbedder(num_classes=args.num_classes, embed_dim=args.embed_dim,
                                   cond_drop_rate=args.cond_drop_rate)
    else:
        raise NotImplementedError(f"Condition model {args.condition_model} is not implemented")

    # Create Optimizer
    logger.info("Creating optimizer")
    # TODO: support faster optimizer
    trainable_params = list(var.parameters())
    if cond_model is not None:
        trainable_params += list(cond_model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    # Compute max_train_steps
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch // accelerator.num_processes

    # Create Learning Rate Scheduler
    logger.info("Creating learning rate scheduler")
    num_warmup_steps = int(args.lr_warmup_steps * args.max_train_steps) if args.lr_warmup_steps < 1.0 else int(
        args.lr_warmup_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
    )

    # Send to accelerator
    var, cond_model, vqvae, optimizer, lr_scheduler, dataloader = accelerator.prepare(var, cond_model, vqvae, optimizer,
                                                                                      lr_scheduler, dataloader)

    # Start tracker
    experiment_config = vars(args)
    accelerator.init_trackers(model_name, config=experiment_config)

    # Start training
    if accelerator.is_main_process:
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    args.completed_steps = 0
    args.starting_epoch = 0

    # TODO: add resume function
    inference(accelerator, var, vqvae, cond_model, np.random.choice(args.num_classes, 4).tolist(), num_samples=1,
              guidance_scale=4.0, top_k=900, top_p=0.95, seed=42)

    # Training
    for epoch in range(args.starting_epoch, args.num_epochs):

        args.epoch = epoch
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        # train epoch
        train_epoch(accelerator, var, vqvae, cond_model, dataloader, optimizer, lr_scheduler, progress_bar, args)

        if epoch % args.val_interval == 0:
            inference(accelerator, var, vqvae, cond_model, np.random.choice(args.num_classes, 4).tolist(),
                      num_samples=1, guidance_scale=4.0, top_k=900, top_p=0.95, seed=42)

        if args.save_interval == 'epoch':
            save_dir = os.path.join(args.project_dir, f"epoch_{args.epoch}")
            os.makedirs(save_dir, exist_ok=True)
            accelerator.save_state(save_dir)

    # end training
    accelerator.end_training()


if __name__ == '__main__':
    main()










