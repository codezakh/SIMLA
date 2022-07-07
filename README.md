[![Conference](http://img.shields.io/badge/ECCV-2022-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Paper](http://img.shields.io/badge/arxiv-cs.CV:2203.14395-B31B1B.svg)](https://arxiv.org/abs/2203.14395)

# SIMLA: Single-Stream Multi-Level Alignment for Vision-Language Pretraining, ECCV 2022 (NEC Labs)

## Pretraining
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain.py --config configs/Pretrain.yaml --output_dir <where to save> 
```

## Retrieval
### Finetuned (COCO)
```bash
python -m torch.distributed.launch --master_port=49770 --nproc_per_node=2 --use_env Retrieval.py \
--config ./configs/Retrieval_coco.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth>
```

### Zero-Shot (Flickr)
```bash
python -m torch.distributed.launch --master_port=47770 --nproc_per_node=2 --use_env zero_shot_retrieval.py --config ./configs/Retrieval_flickr.yaml --output_dir <where to save> --checkpoint <path of .pth checkpoint file> --evaluate
```

### Finetuned (Flickr) 
```bash
python -m torch.distributed.launch --master_port=49770 --nproc_per_node=2 --use_env Retrieval.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth>
```

# Grounding
```bash
python -m torch.distributed.launch --master_port=49121 --nproc_per_node=2 --use_env Grounding.py \
--config ./configs/Grounding.yaml \
--output_dir <path/to/output> \
--gradcam_mode itm \
--block_num 8 \
--checkpoint <path/to/checkpoint.pth> \
```

# VQA
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env VQA.py \
--config ./configs/VQA.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth> 
```
# NLVR
## Pretraining
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain_NLVR.py \
--config ./configs/NLVR_Pretrain.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth> 
```
## Finetuning
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env NLVR.py \
--config ./configs/NLVR.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth>
```
# SNLI-VE
```bash
python -m torch.distributed.launch --master_port=47770 --nproc_per_node=2 \
--use_env VE.py \
--config ./configs/VE.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth>
```