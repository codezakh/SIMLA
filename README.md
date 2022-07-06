# SIMLA
## Pretraining
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain.py --config configs/Pretrain.yaml --output_dir /net/acadia10a/data/zkhan/ALBEF-checkpoints/trash
```