# MFKD: Multi-Faceted Knowledge Distillation

Official implementation of **MFKD**, which transfers instance representation,
instance-class association, and class structure knowledge through three
complementary modules:

* **Instance Representation Distillation (IRD)** (`L_IRD`) contrasts each frozen
  teacher feature with its matched student feature and all other in-batch student
  features, including those from the same class.
* **Target-Class Confidence Distillation (TCCD)** (`L_TCCD`) matches the binary
  distribution of target-class confidence versus all remaining classes.
* **Inter-Class Relational Distillation (ICRD)** (`L_ICRD`) maintains class-conditional
  teacher prediction prototypes by EMA and aligns projected non-target logits.

Experiments cover image classification (CIFAR-100, ImageNet-1K), object detection and
instance segmentation (MS COCO).

## Repository structure

```text
MFKD/                         # repository directory; method name is MFKD
├── configs/                     # classification configs (CIFAR-100, ImageNet-1K)
│   ├── cifar100/
│   │   ├── mfkd/                # 20 MFKD teacher–student pairs
│   │   ├── dkd/                 # 17 DKD configs, for comparison
│   │   └── *.yaml               # vanilla, kd, at, ofd, rkd, crd, reviewkd, lskd, ...
│   └── imagenet/r34_r18/, r50_mv1/
├── mdistiller/
│   ├── distillers/MFKD.py       # >>> the method itself <<<
│   ├── models/                  # CIFAR / ImageNet model zoo
│   ├── dataset/                 # CIFAR-100 and ImageNet data loaders
│   └── engine/                  # cfg.py (hyperparameters), trainer.py
├── tools/
│   ├── train.py                 # classification entry point
│   ├── eval.py                  # accuracy of a checkpoint
│   └── visualizations/          # tsne.py, correlation.py, gradcam.py
└── detection/                   # MS COCO detection & instance segmentation
    ├── configs/MFKD/            # MFKD detection configs
    ├── model/                   # Detectron2 model + the RoI-head MFKD loss
    └── README.md                # >>> detection-specific instructions <<<
```

## Installation

```bash
conda create -n mfkd python=3.8 -y && conda activate mfkd
pip install -r requirements.txt
python setup.py develop
```

Run the installation **from this directory**, so that `develop` registers this working
tree on `sys.path` rather than some other checkout of the toolkit. All entry points
(`tools/*.py`, `tools/visualizations/*.py`, `detection/train_net.py`) additionally put
this directory first on `sys.path`, so they always use the local `mdistiller` even if a
different copy is installed in the environment.

Detection and instance segmentation additionally require
[Detectron2](https://github.com/facebookresearch/detectron2); see
[detection/README.md](detection/README.md).

## Data and pretrained teacher checkpoints

Each dataset has its own root, overridable by an environment variable and defaulting to
a folder next to this README:

* **CIFAR-100** — downloaded automatically on first run into `$MFKD_CIFAR100_ROOT`
  (default `data/cifar100`), which torchvision fills with `cifar-100-python/`.
* **ImageNet-1K** — download from <https://image-net.org/> and lay it out as
  `$MFKD_IMAGENET_ROOT/images/{train,val}` (default `data/imagenet/images`); the
  ImageNet val class names file `classnames.txt` goes in `$MFKD_IMAGENET_ROOT/`,
  i.e. the parent of `images/`.
* **MS COCO** — see [detection/README.md](detection/README.md).

The pretrained CIFAR-100 teachers (the `*_vanilla` checkpoints referenced by the
configs) go into `download_ckpts/cifar_teachers/`. ImageNet and COCO pretrained
weights follow the same convention documented in the [DKD](https://github.com/megvii-research/mdistiller)
and [ReviewKD](https://github.com/dvlab-research/ReviewKD) codebases.

## Training on CIFAR-100

```bash
# MFKD, ResNet32x4 -> ResNet8x4 (the canonical pair)
python3 tools/train.py --cfg configs/cifar100/mfkd/res32x4_res8x4.yaml
```

All 20 released MFKD pairs live in `configs/cifar100/mfkd/`. Any config value can be
overridden from the command line:

```bash
python3 tools/train.py --cfg configs/cifar100/mfkd/res32x4_res8x4.yaml \
    SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1
```

To reproduce the baselines, point `--cfg` at the matching file — for example
`configs/cifar100/kd.yaml`, `configs/cifar100/dkd/res32x4_res8x4.yaml`, or `configs/cifar100/lskd.yaml`.

Checkpoints land in `output/<EXPERIMENT.PROJECT>/<tag>/`, with `student_best`
holding the best student so far. To re-evaluate one:

```bash
python3 tools/eval.py -m resnet8x4 -d cifar100 \
    -c output/cifar100_baselines/mfkd,res32x4,res8x4/student_best
```

## Training on ImageNet-1K

```bash
python3 tools/train.py --cfg configs/imagenet/r34_r18/mfkd.yaml   # ResNet34 -> ResNet18
python3 tools/train.py --cfg configs/imagenet/r50_mv1/mfkd.yaml   # ResNet50 -> MobileNetV1
```

## Training on MS COCO

Detection and instance segmentation use a separate entry point. See
[detection/README.md](detection/README.md) for the full list of commands.

```bash
cd detection
python train_net.py --config-file configs/MFKD/MFKD-R18-R101.yaml      --num-gpus 2
python train_net.py --config-file configs/MFKD/MFKD-R18-R101-Mask.yaml --num-gpus 2
```

## Method overview and code map

The classification method is `mdistiller/distillers/MFKD.py`; its labelled steps
implement the three modules and the final objective:

| Paper                                 | Config keys                          | Code (`MFKD.py`)                     |
| ------------------------------------- | ------------------------------------ | ------------------------------------ |
| Projection head `phi`                 | —                                    | `ProjectionHead`, `self.proj_s`      |
| Embeddings `h^t`, `h^s`               | —                                    | step 1, `F.normalize`                |
| IRD, `L_IRD` — in-batch contrast      | `TAU_F`, `ALPHA`                     | step 1                               |
| `tilde_mu_k` — normalized class means | —                                    | step 2, `mu_tilde`                   |
| Prototype matrix `P` — EMA update     | `PROTO_MOMENTUM`                     | step 2                               |
| `tilde_pi_i` — binary distribution    | —                                    | `_to_binary_target_prob`             |
| TCCD, `L_TCCD` — binary KL            | `TAU_L`                              | step 3                               |
| `P_bar` — row-normalized `P`          | —                                    | step 2, `P_bar`                      |
| ICRD, `L_ICRD` — prototype alignment  | `TAU_L`, `LAMBDA`                    | step 4                               |
| `L_logit`, `L_total`                  | `BETA`, `WARMUP_EPOCHS`              | step 5                               |
| `O_i` — target-class indicator        | —                                    | `_target_mask`                       |

The prototype matrix is `c x c` for classification. In detection, background is treated
as a regular category, so it becomes `(c + 1) x (c + 1)`; the same three modules are
applied to the RoI box head in `detection/model/rcnn.py` (`rcnn_mfkd_loss`).


Python uses `lambda_icrd` for the paper symbol `lambda`. Registered module/buffer names
`proj_s` and `prototypes`, and existing loss dictionary keys, are retained for
checkpoint and logging compatibility. `WARMUP_EPOCHS` and `PROTO_MOMENTUM`
correspond to `E` and `m`; runtime warm-up weight `w_e` is the paper `w^(e)`.

## Hyperparameters

Config keys are named after the symbols of the paper. Classification values vary per
teacher–student pair; the spread across the released configs is:

| Key              | Symbol       | CIFAR-100                         | ImageNet-1K | MS COCO    |
| ---------------- | ------------ | --------------------------------- | ----------- | ---------- |
| `TAU_L`          | `tau_l`      | 4.0                               | 1.0         | 1.0        |
| `TAU_F`          | `tau_f`      | 0.5 / 1.0 / 2.0                   | 2.0         | 1.0        |
| `ALPHA`          | `alpha`      | 10 / 80 / 100 / 120               | 20 / 80     | 1.0        |
| `BETA`           | `beta`       | 0.6 – 1.5                         | 1.0         | 1.0        |
| `LAMBDA`         | `lambda`     | 2.0 / 6.0 / 8.0                   | 0.5 / 2.0   | 0.25       |
| `WARMUP_EPOCHS`  | `E`          | 20                                | 5           | 1          |
| `PROTO_MOMENTUM` | `m`          | 0.99                              | 0.99        | 0.99       |

`CE_WEIGHT` (the coefficient of `L_sup`) is 1.0 everywhere. The MS COCO column lists the
MFKD defaults in `detection/model/config.py`, which the detection configs use as-is.

## Adding your own distillation method

1. Create `mdistiller/distillers/MyDistiller.py` and define the distiller:

   ```python
   from ._base import Distiller

   class MyDistiller(Distiller):
       def __init__(self, student, teacher, cfg):
           super(MyDistiller, self).__init__(student, teacher)
           self.hyper1 = cfg.MyDistiller.HYPER1

       def forward_train(self, image, target, **kwargs):
           # return the output logits and a dict of losses
           ...
       # override get_learnable_parameters if you add nn modules for distillation
       # override get_extra_parameters to report the extra parameter cost
   ```

2. Register it in `distiller_dict` in `mdistiller/distillers/__init__.py`.
3. Register the corresponding hyperparameters in `mdistiller/engine/cfg.py`.
4. Add a config file under `configs/` and run `tools/train.py`.

## Visualizations

```bash
# t-SNE of a CIFAR-100 model's pooled features
python3 tools/visualizations/tsne.py -m wrn_40_2 -c <checkpoint>

# teacher/student class-level correlation matrix
python3 tools/visualizations/correlation.py

# GradCAM comparison of teacher, vanilla student, KD, DKD and MFKD
python3 tools/visualizations/gradcam.py --num 4 --device cuda:0
```

`gradcam.py` reads its checkpoints from `$MFKD_OUTPUT_ROOT`
(default `output/imagenet_baselines`) and `tsne.py` from `$MFKD_CKPT_ROOT`
(default `download_ckpts/cifar_teachers`); both can also be passed explicitly.

## Citation

The paper is currently under double-blind review, so the author list is omitted here
and will be filled in for the camera-ready version.

```BibTeX
@misc{mfkd,
      title={MFKD: Multi-Faceted Knowledge Distillation},
      year={2027},
}
```

## Acknowledgement

The classification codebase is built on [DKD / mdistiller](https://github.com/megvii-research/mdistiller),
and the detection and instance segmentation codebase on
[ReviewKD](https://github.com/dvlab-research/ReviewKD). We thank the authors of both
for making their code available.
