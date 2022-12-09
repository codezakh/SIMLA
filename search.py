import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.simla.retrieval import SIMLA
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


@torch.no_grad()
def search_with_simla(
    model,
    data_loader,
    tokenizer,
    device,
    config):

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation...")
    start_time = time.time()

    data_loader.dataset.text = ["A child sitting at a restaurant table holding a paper mask against his face."]
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 1 #256
    text_feats = []
    text_embeds = []
    text_atts = []
    text_inputs = []
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        text_inputs.append(text_input)
        text_output = model.text_encoder(
            text_input.input_ids, attention_mask=text_input.attention_mask, mode="text"
        )
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_tokens = torch.cat([_.input_ids for _ in text_inputs], dim=0)


    cache = Path(config["cache"])
    cache.mkdir(parents=True, exist_ok=True)

    if (cache / "image_feats.pt").exists() and (cache / "image_embeds.pt").exists():
        print("Loading image features from cache...")
        image_feats = torch.load(cache / "image_feats.pt")
        image_embeds = torch.load(cache / "image_embeds.pt")
    else:
        print("Cache not found. Computing image features...")
        image_feats = []
        image_embeds = []
        for image, img_id in tqdm(data_loader):
            image = image.to(device)
            image_feat = model.visual_encoder(image)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat)
            image_embeds.append(image_embed)

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        torch.save(image_feats, cache / "image_feats.pt")
        torch.save(image_embeds, cache / "image_embeds.pt")
        print('Cached image features at {}'.format(cache))

    sims_matrix = image_embeds @ text_embeds.t()

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):

        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            device
        )
        output = model.text_encoder(
            text_tokens[start + i].repeat(config["k_test"], 1),
            attention_mask=text_atts[start + i].repeat(config["k_test"], 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode="multimodal",
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    import ipdb; ipdb.set_trace()

    return score_matrix_t2i.cpu().numpy()

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset("re", config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [
            None,
            None,
        ]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[config["batch_size_train"]] + [config["batch_size_test"]] * 2,
        num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = SIMLA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    # Delete the .idx_queue from the model, we don't need it for the
    # zero-shot evaluation.
    del model.idx_queue

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )
        state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped

        for key in list(state_dict.keys()):
            if "bert" in key:
                encoder_key = key.replace("bert.", "")
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]

        required_keys = model.state_dict().keys()
        state_dict = {k: v for k, v in state_dict.items() if k in required_keys}
        msg = model.load_state_dict(state_dict, strict=True)

        print("load checkpoint from %s" % args.checkpoint)
        print(msg)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    _, _ = search_with_simla(
        model_without_ddp, test_loader, tokenizer, device, config
    )

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print("Time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Retrieval_flickr.yaml")
    parser.add_argument("--output_dir", default="output/Retrieval_flickr")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
