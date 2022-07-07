import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
import hydra

from models.simla.pretrain import SIMLA
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models.discrete_vae import Dalle_VAE

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from dataset.utils import collate_safe


def train(
    model,
    data_loader,
    optimizer,
    tokenizer,
    epoch,
    warmup_steps,
    device,
    scheduler,
    config,
    image_tokenizer,
    wandb_logger=None,
):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_mlm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_ita", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_itm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_mim", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_pseudo", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (
        image,
        text,
        image_for_tokenization,
        masked_visual_token_positions,
    ) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)

        text_input = tokenizer(
            text, padding="longest", truncation=True, max_length=25, return_tensors="pt"
        ).to(device)
        image_for_tokenization = image_for_tokenization.to(device)
        masked_visual_token_positions = masked_visual_token_positions.to(device)

        if epoch > 0:
            alpha = config["alpha"]
        else:
            alpha = config["alpha"] * min(1, i / len(data_loader))

        with torch.no_grad():
            visual_token_ids = image_tokenizer.get_codebook_indices(
                image_for_tokenization
            ).flatten(1)
            masked_visual_token_pos = masked_visual_token_positions.flatten(1).to(
                torch.bool
            )
            masked_visual_tok_labels = visual_token_ids[masked_visual_token_pos]

        loss_mlm, loss_ita, loss_itm, loss_mim, loss_pseudo = model(
            image,
            text_input,
            visual_token_ids=visual_token_ids,
            masked_visual_token_pos=masked_visual_token_pos,
            masked_visual_tok_labels=masked_visual_tok_labels,
            alpha=alpha,
        )

        loss = loss_mlm + loss_itm + loss_ita + loss_mim + loss_pseudo

        loss.backward()
        grad_norm = utils.calculate_gradient_norm(model)
        optimizer.step()

        if i % print_freq == 0:
            if utils.is_main_process() and wandb_logger:
                wandb_logger.log(
                    data={
                        "loss_itm": loss_itm.item(),
                        "loss_ita": loss_ita.item(),
                        "loss_mlm": loss_mlm.item(),
                        "loss_mim": loss_mim.item(),
                        "loss_pseudo": loss_pseudo.item(),
                        "grad_norm": grad_norm,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_mim=loss_mim.item())
        metric_logger.update(loss_pseudo=loss_pseudo.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]

    #### Dataset ####
    print("Creating dataset")
    datasets = [create_dataset("pretrain", config)]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    data_loader = create_loader(
        datasets,
        samplers,
        batch_size=[config["batch_size"]],
        num_workers=[4],
        is_trains=[True],
        collate_fns=[collate_safe],
    )[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = SIMLA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    disable_wandb = config.get("disable_wandb", False)  # Enable by default.
    if utils.is_main_process() and not disable_wandb:
        print("Is main process, creating W&B logger.")
        wandb_logger = wandb.init(
            project="vision-language-alignment", entity="zakh", config=config
        )
    else:
        wandb_logger = None

    model = model.to(device)

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
        else:
            pos_embed_reshaped = interpolate_pos_embed(
                state_dict["visual_encoder.pos_embed"], model.visual_encoder
            )
            m_pos_embed_reshaped = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
            )
            state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
            state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped
        model.load_state_dict(state_dict)
        print("load checkpoint from %s" % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Loading image tokenizer.")
    image_tokenizer = Dalle_VAE(config["image_res"])
    image_tokenizer.load_model(model_dir=config["image_tokenizer_path"], device=device)
    print(f'Loaded image tokenizer from {config["image_tokenizer_path"]}')

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(
            model,
            data_loader,
            optimizer,
            tokenizer,
            epoch,
            warmup_steps,
            device,
            lr_scheduler,
            config,
            image_tokenizer,
            wandb_logger=wandb_logger,
        )
        if utils.is_main_process():
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(
                save_obj, os.path.join(args.output_dir, "checkpoint_%02d.pth" % epoch)
            )

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Pretrain.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--output_dir", default="Pretrain/")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--overrides", nargs="+", default=[])
    args = parser.parse_args()

    # This is a hack for now to let us use Hydra to override config options
    # without needing to change the run commands substantially. This does break
    # Hydra's ability to compose configurations nicely, but that can be fixed
    # later.
    config_path = Path(args.config)
    with hydra.initialize(config_path=str(config_path.parent)):
        config = hydra.compose(config_name=config_path.stem, overrides=args.overrides)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(
        OmegaConf.to_object(config),
        open(os.path.join(args.output_dir, "config.yaml"), "w"),
    )

    main(args, config)
