import os
import sys

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

# Make this repository's own mdistiller win over any copy installed in the
# environment; without this, running this script resolves `mdistiller` to
# whatever the active environment happens to have installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from mdistiller.models import cifar_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint
from mdistiller.engine.cfg import CFG as cfg

# visualize t-SNE
def get_tsne(stu, mpath):
    cfg.defrost()
    cfg.DISTILLER.STUDENT = stu
    cfg.DATASET.TYPE = 'cifar100'
    cfg.freeze()
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    model = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
    model.load_state_dict(load_checkpoint(mpath)["model"])
    all_features, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(val_loader)):
            outputs, features = model(data)
            all_features.append(features['pooled_feat'].data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)

    tsne = TSNE()
    all_features = tsne.fit_transform(all_features)
    plot_features(all_features, all_labels, num_classes, 'teacher_tsne')

def plot_features(features, labels, num_classes, save_name=None):
    colors = ['C' + str(i) for i in range(num_classes)]
    plt.figure(figsize=(6, 6))
    for l in range(num_classes):
        plt.scatter(
            features[labels == l, 0],
            features[labels == l, 1],
            c=colors[l], s=1, alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    if save_name:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"t-SNE figure saved to: {save_path}")
    plt.show()


def _default_ckpt(student):
    """Default teacher checkpoint shipped with the CIFAR-100 model zoo."""
    root = os.environ.get(
        "MFKD_CKPT_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "download_ckpts", "cifar_teachers"),
    )
    return os.path.join(root, f"{student}_vanilla", "ckpt_epoch_240.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="t-SNE of a CIFAR-100 model's pooled features")
    parser.add_argument("-m", "--model", type=str, default="wrn_40_2",
                        help="student architecture name, e.g. wrn_40_2, resnet32x4")
    parser.add_argument("-c", "--ckpt", type=str, default=None,
                        help="checkpoint path; defaults to the model-zoo teacher weights")
    args = parser.parse_args()

    get_tsne(args.model, args.ckpt or _default_ckpt(args.model))
