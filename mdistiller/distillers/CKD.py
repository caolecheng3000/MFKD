import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller



def ckd_loss(logits_teacher , logits_student, temperature=1.0):
    B = logits_teacher.shape[0]
    device = logits_teacher.device

    # normalize
    logits_teacher = F.normalize(logits_teacher, dim=1)
    logits_student = F.normalize(logits_student, dim=1)

    # cosine similarity as logits
    similarity = logits_teacher @ logits_student.t()

    target = torch.arange(B).to(device)
    contrastive_loss = F.cross_entropy(similarity / temperature, target)

    return contrastive_loss



def contrastive_loss(logits_teacher, logits_student, temperature):

    B = logits_teacher.shape[0]
    device = logits_teacher.device

    logits_teacher = F.normalize(logits_teacher, dim=1) #[B, C]
    logits_student = F.normalize(logits_student, dim=1) #[B, C]

    # cosine similarity as logits
    logits_per_teacher = logits_teacher @ logits_student.t()  # Each row is derived from 1 teacher and B students

    target = torch.arange(B).to(device)
    contrastive_loss = F.cross_entropy(logits_per_teacher / temperature, target)
  
    return contrastive_loss


class CKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(CKD, self).__init__(student, teacher)

        self.temperature = cfg.CKD.TEMPERATURE
        self.ce_loss_weight = cfg.CKD.CE_WEIGHT
        self.cl_loss_weight = cfg.CKD.CL_WEIGHT


    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_ckd = self.cl_loss_weight * contrastive_loss(logits_teacher, logits_student, temperature=self.temperature)              
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_ckd,
        }

        return logits_student, losses_dict
