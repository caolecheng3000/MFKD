import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def logit_standardization(logits, eps=1e-7):
    """Logit Standardization (CVPR 2024): z' = (z - mean(z)) / std(z).

    Removes the scale and shift of logits to make teacher-student
    distributions directly comparable, preventing temperature from being
    dominated by logit magnitude differences.
    """
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (std + eps)


def lskd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    """DKD loss with Logit Standardization pre-processing."""
    # Standardize both teacher and student logits
    logits_student = logit_standardization(logits_student)
    logits_teacher = logit_standardization(logits_teacher)

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    # TCKD: target class binary distillation
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student + 1e-8)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
        * (temperature ** 2)
    )

    # NCKD: non-target class dark knowledge
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
        * (temperature ** 2)
    )

    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class LSKD(Distiller):
    """Logit Standardization Knowledge Distillation (CVPR 2024 Highlight).

    Applies per-sample logit standardization (zero-mean, unit-variance)
    before DKD to remove scale/shift discrepancies between teacher
    and student logit distributions.
    """

    def __init__(self, student, teacher, cfg):
        super(LSKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.LSKD.CE_WEIGHT
        self.alpha = cfg.LSKD.ALPHA
        self.beta = cfg.LSKD.BETA
        self.temperature = cfg.LSKD.T
        self.warmup = cfg.LSKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * lskd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
