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
from dataset import create_dataset 
from scheduler import create_scheduler
from optim import create_optimizer

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def create_loader(dataset, batch_size, num_workers, collate_fn):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        self.extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        self.images = [x for x in self.folder_path.iterdir() if x.suffix.lower() in self.extensions]

    def __getitem__(self, index):
        image_path = os.path.join(self.folder_path, self.images[index])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

@torch.no_grad()
def search_with_simla(
    model,
    data_loader,
    tokenizer,
    device,
    config,
    search_query):

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation...")
    start_time = time.time()

    data_loader.dataset.text = [""]
    texts = [search_query] 
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
        (len(texts), len(data_loader.dataset.images)), -100.0
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

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_t2i.cpu().numpy()

def main(args, config):
    # utils.init_distributed_mode(args)

    device = torch.device(args.device)

    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                )
        ]
    )

    test_dataset = ImageFolderDataset('/net/acadia10a/data/zkhan/flickr30k/flickr30k-images/', transform=test_transform)

    test_loader = create_loader(test_dataset, config['batch_size_test'], 4, None)

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
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    search_query = "A child sitting at a restaurant table holding a paper mask against his face."
    score_matrix_t2i = search_with_simla(
        model_without_ddp, test_loader, tokenizer, device, config, search_query 
    )
    print(score_matrix_t2i.argmax())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Search.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    main(args, config)
