# -*- coding: utf-8 -*-
"""GradCAM comparison on ImageNet-1K.

For one (or several) ImageNet images, this script shows the GradCAM of five
models on the class predicted by the teacher:

  1. Teacher : ResNet34 (ImageNet pretrained)
  2. Student : ResNet18 (ImageNet pretrained, vanilla)
  3. KD      : ResNet18 distilled with KD
  4. DKD     : ResNet18 distilled with DKD
  5. MFKD    : ResNet18 distilled with MFKD

Usage:
  # draw 4 random ImageNet val images
  python tools/visualizations/gradcam.py --num 4 --seed 10 --device cuda:0

  # a specific image
  python tools/visualizations/gradcam.py --image_path /path/to/image.jpg --device cuda:0

The three distillation checkpoints default to the layout produced by
tools/train.py (output/imagenet_baselines/<tag>/student_best) and can be
overridden with --kd_ckpt / --dkd_ckpt / --mfkd_ckpt.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")  # save PNGs headless, without requiring a GUI backend
import matplotlib.pyplot as plt

# Make this repository's own mdistiller win over any copy installed in the
# environment; without this, running this script resolves `mdistiller` to
# whatever the active environment happens to have installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from mdistiller.models import imagenet_model_dict
from mdistiller.dataset.imagenet import data_folder as IMAGENET_DATA_FOLDER
from mdistiller.engine.utils import load_checkpoint


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DISP_SIZE = 224  # common size for display and for upsampling the CAM

# Default checkpoint locations (ImageNet, ResNet34 teacher -> ResNet18 student).
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "MFKD_OUTPUT_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output", "imagenet_baselines"),
)
KD_CKPT = os.path.join(DEFAULT_OUTPUT_ROOT, "kd,res34,res18", "student_best")
DKD_CKPT = os.path.join(DEFAULT_OUTPUT_ROOT, "dkd,res34,res18", "student_best")
MFKD_CKPT = os.path.join(DEFAULT_OUTPUT_ROOT, "mfkd,res34,res18", "student_best")


# --------------------------------------------------------------------------- #
# GradCAM
# --------------------------------------------------------------------------- #
class GradCAM:
    """Standard GradCAM on the last convolutional stage (layer4, else layer3)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input_, output):
        self.activations = output[0] if isinstance(output, tuple) else output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, target_class=None):
        """Return (cam [1,1,h,w], logits [1,C], pred_idx)."""
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(x)[0]
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        activations = self.activations.detach()              # [1, C, h, w]
        gradients = self.gradients.detach()                  # [1, C, h, w]
        weights = gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # [1, 1, h, w]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, logits.detach(), target_class


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _pick_target_layer(model):
    """Prefer layer4, fall back to layer3."""
    for name in ("layer4", "layer3"):
        layer = getattr(model, name, None)
        if layer is not None:
            return layer
    raise ValueError("no usable convolutional stage found (layer4/layer3)")


def build_imagenet_model(arch, ckpt_path=None):
    """Build an ImageNet model.

    ckpt_path=None loads the torchvision ImageNet weights (teacher / vanilla
    student); otherwise the given student_best checkpoint is loaded on top of
    the plain ResNet architecture (KD / DKD / MFKD students).
    """
    model = imagenet_model_dict[arch](pretrained=(ckpt_path is None))
    if ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        state = load_checkpoint(ckpt_path)
        state = state["model"] if "model" in state else state
        model.load_state_dict(state)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Data and preprocessing
# --------------------------------------------------------------------------- #
def _normalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (tensor - mean) / std


def prepare_input(img_tensor):
    """[3,H,W] in [0,1] -> [1,3,224,224], following Resize(256)+CenterCrop(224)."""
    x = F.interpolate(
        img_tensor.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
    )
    start = (256 - DISP_SIZE) // 2
    x = x[:, :, start:start + DISP_SIZE, start:start + DISP_SIZE]
    x = _normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x


def make_display(img_tensor):
    """Return a 224x224 [H,W,3] numpy array in [0,1], for display and CAM overlay."""
    x = F.interpolate(
        img_tensor.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
    )
    start = (256 - DISP_SIZE) // 2
    x = x[:, :, start:start + DISP_SIZE, start:start + DISP_SIZE]
    return x[0].permute(1, 2, 0).numpy()


def _load_imagenet_names(path):
    """Parse classnames.txt (one "<wnid> <name>" per line) into wnid -> name."""
    wnid_to_name = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, name = line.split(" ", 1)
            wnid_to_name[wnid] = name
    return wnid_to_name


def load_images(args):
    """Return (images, labels, class_names).

    With --image_path the single image is returned with label=None; otherwise
    ``args.num`` images are drawn at random from the ImageNet val split and the
    ground-truth labels are kept.
    """
    val_folder = os.path.join(IMAGENET_DATA_FOLDER, "val")
    classnames_path = os.path.join(os.path.dirname(IMAGENET_DATA_FOLDER), "classnames.txt")
    wnid_to_name = _load_imagenet_names(classnames_path)

    # Use the val folder to recover the torchvision class index order, which is
    # only needed to display readable prediction names.
    ds = datasets.ImageFolder(val_folder, transform=transforms.ToTensor())
    class_names = [wnid_to_name[c] for c in ds.classes]

    if args.image_path:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"image not found: {args.image_path}")
        pil_img = Image.open(args.image_path).convert("RGB")
        img = transforms.ToTensor()(pil_img)
        return [img], [None], class_names

    seed = args.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")  # fresh sample on every run
    rng = np.random.default_rng(seed)
    print(f"[INFO] random seed: {seed} (use --seed {seed} to reproduce this draw)")
    indices = rng.choice(len(ds), size=args.num, replace=False).tolist()
    images, labels = [], []
    for idx in indices:
        img, label = ds[idx]
        images.append(img)
        labels.append(label)
    return images, labels, class_names


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _overlay(img_rgb, cam, alpha=0.5):
    """Overlay a [0,1] CAM on a [0,1] RGB image using the jet colormap."""
    cmap = plt.get_cmap("jet")
    cam_rgb = cmap(cam)[..., :3]
    return (1 - alpha) * img_rgb + alpha * cam_rgb


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device(args.device)

    # The five models under comparison: (display name, architecture, checkpoint or None)
    model_specs = [
        (f"Teacher {args.teacher}", args.teacher, None),
        (f"Student {args.student}", args.student, None),
        ("KD", args.student, args.kd_ckpt),
        ("DKD", args.student, args.dkd_ckpt),
        ("MFKD", args.student, args.mfkd_ckpt),
    ]

    gcams = []
    for label, arch, ckpt in model_specs:
        model = build_imagenet_model(arch, ckpt).to(device)
        layer = _pick_target_layer(model)
        gcams.append((label, GradCAM(model, layer)))
        src = "pretrained" if ckpt is None else ckpt
        print(f"[INFO] loaded {label}: {arch} ({src})")

    images, labels, class_names = load_images(args)
    print(f"[INFO] number of images: {len(images)}")

    for i, (img, gt_label) in enumerate(zip(images, labels)):
        x = prepare_input(img).to(device)
        img_np = make_display(img)

        # Use the teacher's prediction as the shared target class so that the
        # five models are compared on identical evidence.
        cam_t, _, pred_t = gcams[0][1](x)
        target_class = pred_t
        ref_name = class_names[target_class]
        gt_name = class_names[gt_label] if gt_label is not None else ref_name

        cams = [cam_t]
        preds = [pred_t]
        for _, gcam in gcams[1:]:
            cam, _, pred = gcam(x, target_class=target_class)
            cams.append(cam)
            preds.append(pred)

        # Layout: the input image followed by the GradCAM of each model.
        fig, axes = plt.subplots(1, 1 + len(gcams), figsize=(19, 3.6))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input\n{gt_name}", fontsize=10)
        axes[0].axis("off")

        for j, ((label, _), cam) in enumerate(zip(gcams, cams)):
            cam_up = F.interpolate(
                cam, size=(DISP_SIZE, DISP_SIZE), mode="bilinear", align_corners=False
            )
            cam_np = cam_up[0, 0].cpu().numpy()
            axes[j + 1].imshow(_overlay(img_np, cam_np))
            axes[j + 1].set_title(f"{label}\n{class_names[preds[j]]}", fontsize=10)
            axes[j + 1].axis("off")

        fig.suptitle(f"GradCAM @ class '{ref_name}'", y=1.02, fontsize=13)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"gradcam_5models_{i}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] saved {out_path}")

    print("[INFO] done.")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="GradCAM comparison on ImageNet-1K (5 models)"
    )
    parser.add_argument("--teacher", type=str, default="ResNet34")
    parser.add_argument("--student", type=str, default="ResNet18")
    parser.add_argument("--kd_ckpt", type=str, default=KD_CKPT)
    parser.add_argument("--dkd_ckpt", type=str, default=DKD_CKPT)
    parser.add_argument("--mfkd_ckpt", type=str, default=MFKD_CKPT)
    parser.add_argument("--image_path", type=str, default=None,
                        help="a specific image; if omitted, sample from ImageNet val")
    parser.add_argument("--num", type=int, default=4,
                        help="number of images to sample (ignored with --image_path)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; None means a fresh draw on every run")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default=here)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
