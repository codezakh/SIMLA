[![Conference](http://img.shields.io/badge/ECCV-2022-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Paper](http://img.shields.io/badge/arxiv-cs.CV:2203.14395-B31B1B.svg)](https://arxiv.org/abs/2203.14395)

# SIMLA: Single-Stream Multi-Level Alignment for Vision-Language Pretraining, ECCV 2022 (NEC Labs)
This is the official PyTorch implementation of [SIMLA](https://arxiv.org/abs/2203.14395).
The repository is heavily based on [salesforce/ALBEF](https://github.com/salesforce/ALBEF), and supports vision-language pretraining and downstream task finetuning for several tasks.

<img src="teaser.PNG" width="800">

## Setup
### Dependencies
```
conda env create --name simla --file environment.yaml
```
### Data
See individual sections below for instructions.

## Checkpoints

- [pretrained on 4m images](https://drive.google.com/file/d/1fALNNEJM6bNkVhY4vDlT8JnSWhkQ1Sop/view?usp=sharing)
    - Use this one if you want to finetune the model for another downstream VL task, like VQA. 
- [finetuned on COCO](https://drive.google.com/file/d/16S99bJ-tCAYX0IcN6FM-W-yzD59F2UdV/view?usp=sharing)
    - Use this one for retrieval tasks.

The checkpoints are around 3GB, and contain the optimizer state and everything else needed to resume training.



## Pretraining
Downloading these exact datasets is unnecessary - the pretraining only requires image text pairs, so any image-text pair dataset will do.
- Download COCO from the [official website](https://cocodataset.org/#download) (use COCO2014, download it all).
- Download SBU captions using [Huggingface](https://huggingface.co/datasets/sbu_captions).
- Download Conceptual Captions using [Huggingface](https://huggingface.co/datasets/conceptual_captions).

1. Download the weights of DALL-E's D-VAE ([encoder](https://cdn.openai.com/dall-e/encoder.pkl), [decoder](https://cdn.openai.com/dall-e/decoder.pkl)), and place them in a folder.
2. Edit `configs/Pretrain.yaml` and change `image_tokenizer_path: /net/acadia10a/data/zkhan/dall-e-tokenizer-weights` to the folder where you downloaded the dall-e tokenizer weights.
3. Generate the pretraining JSON. You can download an example from [ALBEF](https://github.com/salesforce/ALBEF#download).
    - The JSON is a list of dictionaries, one for each image: `{'image': '/absolute/path/to/image', 'caption': 'the caption of image'}`.
    - We made a JSON file for each dataset we used (COCO, SBU, CC3M), but you can just have one file for all the image text-pairs.
4. Edit `configs/Pretrain.yaml` and point it to your JSON, so `train_file: /path/to/your/pretraining.json`.
5. Run the command below.
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain.py --config configs/Pretrain.yaml --output_dir <where to save> 
```
Pretraining on 8x A100s with a batch size of 64 for 30 epochs takes roughly 7 days, and uses about 73GB of GPU memory per GPU.
If you're using A100s or A6000s, you may need to run `export NCCL_P2P_DISABLE=1` in the shell before training.

## Image Text Retrieval
1. Download the JSON files for finetuning [here](https://drive.google.com/file/d/1iJ5XzqImROIBBsoiz8R7ayuML6kXoHRr/view?usp=sharing).
2. Next, download the COCO2017 train images and val images from the [official website](https://cocodataset.org/#download), and move all the images into one directory.

### Finetuned (COCO)
Edit `train_file`, `val_file` and `test_file` in `configs/Retrieval_coco.yaml` to point to their respective JSON files you downloaded in Step 1. 
Note that the test annotations are not public, so we report results on the validation split following previous work.

```bash
python -m torch.distributed.launch --master_port=49770 --nproc_per_node=2 --use_env Retrieval.py \
--config ./configs/Retrieval_coco.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth>
```

### Zero-Shot (Flickr)
Download Flickr30k from [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).
Download the annotations for Flickr30k [here](https://drive.google.com/file/d/1AAIdL4E-9-7algjd4DFW42LtXrXWV_z6/view?usp=sharing). 
Edit `train_file`, `val_file` and `test_file` in `configs/Retrieval_flickr.yaml` to point to their respective JSON files you downloaded in Step 1. 
We do not use the validation split, so that key can be set to the name of the train or test file.

```bash
python -m torch.distributed.launch --master_port=47770 --nproc_per_node=2 --use_env zero_shot_retrieval.py --config ./configs/Retrieval_flickr.yaml --output_dir <where to save> --checkpoint <path of .pth checkpoint file> --evaluate
```

### Finetuned (Flickr) 
Same as the above.
```bash
python -m torch.distributed.launch --master_port=49770 --nproc_per_node=2 --use_env Retrieval.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth>
```

## RefCOCO+ (Visual Grounding) 
```bash
python -m torch.distributed.launch --master_port=49121 --nproc_per_node=2 --use_env Grounding.py \
--config ./configs/Grounding.yaml \
--output_dir <path/to/output> \
--gradcam_mode itm \
--block_num 8 \
--checkpoint <path/to/checkpoint.pth> \
```

## VQA (Visual Question Answering)
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env VQA.py \
--config ./configs/VQA.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth> 
```
## NLVR (Natural Language Visual Reasoning)
### Pretraining
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain_NLVR.py \
--config ./configs/NLVR_Pretrain.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth> 
```
### Finetuning
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env NLVR.py \
--config ./configs/NLVR.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth>
```
## SNLI-VE (Visual Entailment)
```bash
python -m torch.distributed.launch --master_port=47770 --nproc_per_node=2 \
--use_env VE.py \
--config ./configs/VE.yaml \
--output_dir <path/to/output> \
--checkpoint <path/to/checkpoint.pth>
```

## Citation
```
@inproceedings{SIMLA,
      title={Single-Stream Multi-Level Alignment for Vision-Language Pretraining, 
      author={Zaid Khan and Vijay Kumar BG and Xiang Yu and Samuel Schulter and Manmohan Chandraker and Yun Fu},
      year={2022},
      booktitle={ECCV}
}
```