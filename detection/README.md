# COCO object detection and instance segmentation

Built on the [ReviewKD](https://github.com/dvlab-research/ReviewKD) codebase.

## Environment

Tested with:

* 2–4 GPUs (`--num-gpus` below should match the number you actually use)
* python 3.8
* torch 1.11.0 / torchvision 0.12.0
* detectron2 0.6

## Installation

The code is based on Detectron2, which is not on PyPI; install it from source
following <https://github.com/facebookresearch/detectron2>. Pick the Detectron2
revision that matches your torch/torchvision build.

All commands below are run from inside this `detection/` directory, since
`train_net.py` imports the local `model` package. `train_net.py` also puts the
repository root first on `sys.path`, so it uses this checkout's `mdistiller`
regardless of what is installed in the environment.

## Data

Place the [COCO](https://cocodataset.org/#download) 2017 dataset so that it matches the
`coco_2017_train` / `coco_2017_val` registrations used by `configs/Base-Distillation.yaml`:

```text
detection/datasets/coco/
├── annotations/          # instances_train2017.json, instances_val2017.json, ...
├── train2017/
└── val2017/
```

## Pretrained weights

Place the teacher and student weights in `pretrained/`. The pretrained models are
available from the [ReviewKD release page](https://github.com/dvlab-research/ReviewKD/releases/);
they contain both the teacher's and the student's weights. The teacher's weights come
from Detectron2's pretrained detector, and the student's weights are ImageNet
pretrained weights.

Each config loads `MODEL.WEIGHTS` from one of six files, named `<student>-<teacher>`:

| File                          | Teacher        | Student        | Used by                          |
| ----------------------------- | -------------- | -------------- | -------------------------------- |
| `pretrained/r18-r101.pth`     | ResNet-101     | ResNet-18      | `*-R18-R101.yaml`                |
| `pretrained/r18-r101mask.pth` | ResNet-101     | ResNet-18      | `*-R18-R101-Mask.yaml`           |
| `pretrained/r50-r101.pth`     | ResNet-101     | ResNet-50      | `*-R50-R101.yaml`                |
| `pretrained/r50-r101mask.pth` | ResNet-101     | ResNet-50      | `*-R50-R101-Mask.yaml`           |
| `pretrained/mv2-r50.pth`      | ResNet-50      | MobileNetV2    | `*-MV2-R50.yaml`                 |
| `pretrained/mv2-r50mask.pth`  | ResNet-50      | MobileNetV2    | `*-MV2-R50-Mask.yaml`            |

## Training

MFKD (`KD.TYPE: "MFKD"`, see `configs/MFKD/`):

```bash
# Tea: R-101, Stu: R-18
python train_net.py --config-file configs/MFKD/MFKD-R18-R101.yaml --num-gpus 2
python train_net.py --config-file configs/MFKD/MFKD-R18-R101-Mask.yaml --num-gpus 2

# Tea: R-101, Stu: R-50
python train_net.py --config-file configs/MFKD/MFKD-R50-R101.yaml --num-gpus 2
python train_net.py --config-file configs/MFKD/MFKD-R50-R101-Mask.yaml --num-gpus 2

# Tea: R-50, Stu: MV2
python train_net.py --config-file configs/MFKD/MFKD-MV2-R50.yaml --num-gpus 2
python train_net.py --config-file configs/MFKD/MFKD-MV2-R50-Mask.yaml --num-gpus 2
```

The baselines reproduce with the same pattern, swapping the config directory:

```bash
python train_net.py --config-file configs/DKD/DKD-R18-R101.yaml --num-gpus 2
python train_net.py --config-file configs/ReviewKD/ReviewKD-R18-R101.yaml --num-gpus 2
python train_net.py --config-file configs/ReviewDKD/ReviewDKD-R18-R101.yaml --num-gpus 2
```

Checkpoints and inference results are written to `OUTPUT_DIR` (for example
`output/MFKD-R18-R101/`), with the final model saved as `model_final.pth`.

## Evaluation

Results are reported on `coco_2017_val`. To evaluate a trained checkpoint:

```bash
python train_net.py \
    --config-file configs/MFKD/MFKD-R18-R101.yaml \
    --num-gpus 2 --eval-only \
    MODEL.WEIGHTS output/MFKD-R18-R101/model_final.pth
```

## Method overview and code map

MFKD distills both the RoI box-head features and the `c+1`-way box predictions from the
frozen teacher detector; the mask branch keeps the standard supervised loss. The three
modules of the paper are all implemented in `model/rcnn.py`:

| Paper                                    | Code                                                          |
| ---------------------------------------- | ------------------------------------------------------------- |
| IRD, `L_IRD` — in-batch contrast         | `rcnn_mfkd_loss`, step 1                                      |
| `tilde_mu_k`, prototype matrix `P` (EMA) | `rcnn_mfkd_loss`, step 2, `mu_tilde`                          |
| TCCD, `L_TCCD` — binary KL               | `rcnn_mfkd_loss`, step 3                                      |
| ICRD, `L_ICRD` — prototype alignment     | `rcnn_mfkd_loss`, steps 2 and 4                               |
| `L_logit`, `L_total`                     | `rcnn_mfkd_loss`, "Objective"                                 |
| RoI box-head features + logits           | `RCNNKD.forward_pure_roi_head`, `RCNNKD.forward`              |

Background is treated as a regular category, so the prototype matrix `P` is
`(c + 1) x (c + 1)` and background RoIs contribute to `mu_k` of the background class
like any other class; a background RoI's non-target logits are its `c` object-class
scores.


Python uses `lambda_icrd` for the paper symbol `lambda`. The registered buffer name
`prototypes` and existing loss dictionary keys are retained for checkpoint and
logging compatibility. `WARMUP_EPOCHS` and `PROTO_MOMENTUM`
correspond to `E` and `m`; runtime warm-up weight `w_e` is the paper `w^(e)`.

## MFKD hyperparameters

The hyperparameters under `KD.MFKD` follow the notation of the paper and keep the
defaults in `model/config.py`, which match the settings reported for MS COCO
(`alpha = beta = tau_f = tau_l = 1.0`):

| Config key        | Symbol       | Meaning                                  | Default |
| ----------------- | ------------ | ---------------------------------------- | ------- |
| `TAU_L`           | `tau_l`      | temperature of the logit-level softmax   | 1.0     |
| `TAU_F`           | `tau_f`      | temperature of the feature InfoNCE loss  | 1.0     |
| `ALPHA`           | `alpha`      | weight of `L_IRD`                        | 1.0     |
| `BETA`            | `beta`       | weight of `L_logit`                      | 1.0     |
| `LAMBDA`          | `lambda`     | weight of `L_ICRD` inside `L_logit`      | 0.25    |
| `WARMUP_EPOCHS`   | `E`          | number of warm-up epochs                 | 1       |
| `PROTO_MOMENTUM`  | `m`          | EMA momentum of the prototype matrix `P` | 0.99    |

## Acknowledgement

This codebase is built on [ReviewKD](https://github.com/dvlab-research/ReviewKD).
