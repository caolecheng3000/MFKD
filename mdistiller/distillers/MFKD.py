import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def _get_pooled_feature_dim(model):
    """Return the dimension of the pooled feature that enters the classifier.

    Supports: fc (CIFAR ResNet, WRN), linear (ResNet v2, ShuffleNet),
              classifier (VGG, MobileNetV2).
    """
    for attr in ["fc", "linear", "classifier"]:
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Linear):
                return head.in_features
            if isinstance(head, nn.Sequential):
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
    raise AttributeError(
        f"Cannot find classification head (fc/linear/classifier) in {type(model).__name__}. "
        f"Available attributes: {[a for a in dir(model) if not a.startswith('_')]}"
    )


def _get_num_classes(model):
    """Return the number of output classes, read off the classification head."""
    for attr in ["fc", "linear", "classifier"]:
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Linear):
                return head.out_features
            if isinstance(head, nn.Sequential):
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
    raise AttributeError(
        f"Cannot find classification head to infer num_classes in {type(model).__name__}."
    )


class ProjectionHead(nn.Module):
    """The projection head phi: R^{d_s} -> R^{d_t} (a single linear layer).

    It is attached to the student side only and is used during training only,
    so that teacher and student embeddings can be compared on a common sphere.
    """

    def __init__(self, dim_in, dim_out):
        super(ProjectionHead, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)


def _target_mask(logits, target):
    """Return a batch of one-hot label vectors O_i as a boolean mask."""
    target = target.reshape(-1)
    return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()


def _to_binary_target_prob(prob, target):
    """Compress a c-dimensional distribution into the binary (target, non-target) form."""
    target_prob = prob.gather(1, target.unsqueeze(1))
    other_prob = prob.sum(dim=1, keepdims=True) - target_prob
    return torch.cat([target_prob, other_prob], dim=1)


class MFKD(Distiller):
    """Multi-Faceted Knowledge Distillation.

    Three complementary modules, matching the current paper:
      IRD  (L_IRD)  -- instance representation knowledge through in-batch contrast
      TCCD (L_TCCD) -- instance-class association through binary confidence matching
      ICRD (L_ICRD) -- class structure knowledge through prototype-guided alignment

    The 'loss_kd' entry contains alpha * L_IRD + beta * L_logit,
    where L_logit = w^(e) * (L_TCCD + lambda * L_ICRD).
    The supervised classification loss L_sup is returned under 'loss_ce'.
    Python uses lambda_icrd for the paper's lambda because lambda is a keyword.
    Registered proj_s/prototypes names retain the checkpoint keys for phi/P.
    """

    def __init__(self, student, teacher, cfg):
        super(MFKD, self).__init__(student, teacher)
        self.ce_weight = cfg.MFKD.CE_WEIGHT
        self.tau_l = cfg.MFKD.TAU_L
        self.tau_f = cfg.MFKD.TAU_F
        self.alpha = cfg.MFKD.ALPHA
        self.beta = cfg.MFKD.BETA
        self.lambda_icrd = cfg.MFKD.LAMBDA
        self.E = cfg.MFKD.WARMUP_EPOCHS
        self.m = cfg.MFKD.PROTO_MOMENTUM

        self.d_t = _get_pooled_feature_dim(self.teacher)
        self.d_s = _get_pooled_feature_dim(self.student)

        # phi; retain proj_s as the registered module name for checkpoint compatibility.
        self.proj_s = ProjectionHead(self.d_s, self.d_t)

        self.c = _get_num_classes(self.teacher)
        # P^(0) = I_c; retain the registered buffer names used by existing checkpoints.
        self.register_buffer("prototypes", torch.eye(self.c))
        self.register_buffer("eye_buffer", torch.eye(self.c))

    def get_learnable_parameters(self):
        params = [v for k, v in self.student.named_parameters()]
        params += [v for k, v in self.proj_s.named_parameters()]
        return params

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.proj_s.parameters())

    def forward_train(self, image, target, **kwargs):
        e = kwargs.get("epoch", 1)
        w_e = min(e / self.E, 1.0)

        l_s, feats_s = self.student(image)
        with torch.no_grad():
            l_t, feats_t = self.teacher(image)

        f_s = feats_s["pooled_feat"]
        f_t = feats_t["pooled_feat"]

        # ---- 1. IRD: in-batch instance representation contrast (L_IRD) ----
        h_s = F.normalize(self.proj_s(f_s), dim=1)
        h_t = F.normalize(f_t, dim=1)

        scaled_s = torch.matmul(h_t, h_s.t()) / self.tau_f  # entries s_ij / tau_f
        exp_scaled_s = torch.exp(scaled_s)
        positive_exp = torch.diag(exp_scaled_s)
        partition_exp = exp_scaled_s.sum(dim=1)  # includes the positive and all negatives
        loss_ird = -torch.log(positive_exp / (partition_exp + 1e-8)).mean()

        # ---- 2. ICRD state: P updated with normalized class means tilde_mu_k ----
        with torch.no_grad():
            pi_t = F.softmax(l_t / self.tau_l, dim=1)  # [B, C] = \Pi^t in the paper, one row \pi_i^t per sample
            O = F.one_hot(target, num_classes=self.c).float()  # O[i, k] = I[y_i = k]
            class_counts = O.sum(dim=0)  # [C]
            class_sums = O.t() @ pi_t  # [C, C]
            present_classes = class_counts > 0
            safe_counts = class_counts.clamp(min=1)
            # Each row is tilde_mu_k, the normalized batch mean from the paper.
            mu_tilde = F.normalize(class_sums / safe_counts.unsqueeze(1), dim=1)
            self.prototypes[present_classes] = (
                self.m * self.prototypes[present_classes]
                + (1 - self.m) * mu_tilde[present_classes]
            )
            # Existing safeguard: mix P slightly with the identity matrix.
            self.prototypes = (1 - 1e-6) * self.prototypes + 1e-6 * self.eye_buffer

        P_bar = F.normalize(self.prototypes, dim=1)  # row-normalized P, i.e. \bar{P}

        # ---- 3. TCCD: binary target-class confidence matching (L_TCCD) ----
        O_mask = _target_mask(l_s, target)
        pi_s_tilde = _to_binary_target_prob(F.softmax(l_s / self.tau_l, dim=1), target)
        pi_t_tilde = _to_binary_target_prob(F.softmax(l_t / self.tau_l, dim=1), target)
        loss_tccd = (
            F.kl_div(torch.log(pi_s_tilde + 1e-8), pi_t_tilde, reduction="none").sum(dim=1)
            * (self.tau_l ** 2)
        ).mean()

        # ---- 4. ICRD: prototype-guided non-target logit alignment (L_ICRD) ----
        u_s = l_s.masked_fill(O_mask, 0)  # u^s: student non-target logits
        u_t = l_t.masked_fill(O_mask, 0)  # u^t: teacher non-target logits
        u_s_hat = u_s @ P_bar.t()  # \hat{U}^s = U^s \bar{P}^T
        u_t_hat = u_t @ P_bar.t()  # \hat{U}^t = U^t \bar{P}^T
        prob_t_projected = F.softmax(u_t_hat / self.tau_l, dim=1)
        log_prob_s_projected = F.log_softmax(u_s_hat / self.tau_l, dim=1)
        loss_icrd = (
            F.kl_div(log_prob_s_projected, prob_t_projected, reduction="none").sum(dim=1) * (self.tau_l ** 2)
        ).mean()

        # ---- 5. Objective ----
        # L_logit = w^(e) * (L_TCCD + lambda * L_ICRD); the warm-up acts only here.
        loss_logit = w_e * (loss_tccd + self.lambda_icrd * loss_icrd)
        loss_mfkd = self.alpha * loss_ird + self.beta * loss_logit

        loss_sup = F.cross_entropy(l_s, target)

        losses_dict = {
            "loss_ce": self.ce_weight * loss_sup,
            "loss_kd": loss_mfkd,
        }

        return l_s, losses_dict
