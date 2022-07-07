# SIMLA
## Pretraining
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain.py --config configs/Pretrain.yaml --output_dir <where to save> 
```

## Retrieval
### Finetuned (COCO)
```bash
python -m torch.distributed.launch --master_port=49770 --nproc_per_node=2 --use_env Retrieval.py \
--config ./configs/Retrieval_coco.yaml \
--output_dir <path/to/output.pth> \
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
--output_dir <path/to/output.pth> \
--checkpoint <path/to/checkpoint.pth>
```

# Grounding
# VQA
# NLVR
# SNLI-VE