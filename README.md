# MicroSight: An Efficient Transformer-based Middleware for Detecting Small Objects

MicroSight is an efficient transformer-based middleware for improving small object detection via multi-perspective feature extraction and collaboration. This repository provides two reference implementations:

- `Mi-DETR/`: DETR-style training code with MicroSight integrated
- `Mi-CODETR/`: Co-DETR/MMDetection-style training code with MicroSight integrated

## Repository Layout

```
code/MicroSight
├─ Mi-DETR/        # DETR-style training/eval entrypoints (main.py)
├─ Mi-CODETR/      # MMDetection-style project (configs/tools/projects)
├─ README.md       # This file
└─ requirements.txt
```

## Environment

This repo contains two independent codebases with different dependency stacks. Use the subproject-specific installation for best reproducibility:

- `Mi-DETR/` is closest to the original DETR repo and uses lightweight PyTorch dependencies.
- `Mi-CODETR/` is based on MMDetection and requires MMCV/MMDetection-compatible versions of PyTorch/CUDA.

## Installation

### Option A: Mi-DETR (DETR-style)

```bash
cd Mi-DETR
pip install -r requirements.txt
```

For COCO evaluation, `pycocotools` is required (already included in `Mi-DETR/requirements.txt`).

### Option B: Mi-CODETR (MMDetection-style)

`Mi-CODETR` follows the upstream Co-DETR/MMDetection workflow. The upstream README documents tested versions and installation details:

- `python=3.7.11, pytorch=1.11.0, cuda=11.3`

Install dependencies and build the project:

```bash
cd Mi-CODETR
pip install -r requirements.txt
pip install -v -e .
```

If you encounter issues installing `mmcv-full`, follow the official MMCV installation instructions matching your PyTorch/CUDA version.

## Data Preparation

### COCO (Mi-DETR)

Download COCO 2017 and organize it as:

```
/path/to/coco
├─ annotations/
├─ train2017/
└─ val2017/
```

Then pass `--coco_path /path/to/coco` to training/evaluation commands.

### COCO (Mi-CODETR)

The default MMDetection-style layout is:

```
Mi-CODETR/data/coco
├─ annotations/
├─ train2017/
└─ val2017/
```

## Training & Evaluation

### Mi-DETR

Single-node multi-GPU training example:

```bash
cd Mi-DETR
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco --output_dir outputs/microsight_detr
```

Evaluation example:

```bash
cd Mi-DETR
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco --resume /path/to/checkpoint.pth --eval
```

### Mi-CODETR

Training with 8 GPUs:

```bash
cd Mi-CODETR
sh tools/dist_train.sh <config.py> 8 <work_dir>
```

Evaluation:

```bash
cd Mi-CODETR
sh tools/dist_test.sh <config.py> <checkpoint.pth> 8 --eval bbox
```

Refer to [Mi-CODETR/README.md](Mi-CODETR/README.md) for the complete upstream training/evaluation workflow and available configs.

## Implementation Notes

For readers who want to locate key components quickly:

- `Mi-DETR/models/transformer.py` contains the main architectural modifications used by MicroSight (e.g., bidirectional LSTM-based global interaction and graph-convolution-based local-range optimization).
- `Mi-CODETR/` follows the MMDetection registry/config patterns; the training entrypoints are in `Mi-CODETR/tools/` and project models are under `Mi-CODETR/projects/`.

## Acknowledgement

The codebases are adapted from:

- DETR: https://github.com/facebookresearch/detr
- Co-DETR: https://github.com/Sense-X/Co-DETR

Please follow their original licenses and third-party notices where applicable.

## Manuscript, Preprint, and Citation

This repository is directly related to a manuscript submitted to *The Visual Computer*. A preprint version has also been released. If you use this code, please cite the manuscript/preprint.

```bibtex
@misc{gu2026microsight_tvc_under_review,
  title   = {Enhancing Small Object Detection: A Transformer-Based Middleware Approach},
  author  = {Gu, Fei and Wang, Zijun and Zhang, Ziheng and Zhang, Zeyang and Wu, Jing},
  year    = {2026},
  note    = {Submitted to The Visual Computer (under review)}
}

@misc{gu2026microsight_preprint,
  title        = {Enhancing Small Object Detection: A Transformer-Based Middleware Approach},
  author       = {Gu, Fei and Wang, Zijun and Zhang, Ziheng and Zhang, Zeyang and Wu, Jing},
  year         = {2026},
  howpublished = {Preprint},
  url          = {TBD},
  note         = {Please replace this entry with the final preprint URL/DOI}
}

@software{microsight_zenodo_18290610,
  title  = {MicroSight},
  author = {Gu, Fei and Wang, Zijun and Zhang, Ziheng and Zhang, Zeyang and Wu, Jing},
  year   = {2026},
  doi    = {10.5281/zenodo.18290610},
  url    = {https://doi.org/10.5281/zenodo.18290610}
}
```
