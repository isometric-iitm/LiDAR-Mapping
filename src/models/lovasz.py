import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 1e-6


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.float().sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / (union + EPS)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes: list[int] | None = None) -> torch.Tensor:
    if probas.ndim != 2:
        raise ValueError(f"Expected 2D probs, got {probas.ndim}")
    C = probas.size(1)
    losses = []
    classes_to_sum = classes if classes is not None else range(C)
    for c in classes_to_sum:
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (probas[:, c] * fg).sort(descending=True).values
        grad = lovasz_grad(fg)
        loss_c = torch.dot(F.relu(errors - 1.0) + 1.0, grad)
        if torch.isnan(loss_c) or torch.isinf(loss_c):
            continue
        losses.append(loss_c)
    if len(losses) == 0:
        return probas[:, 0].sum() * 0.0
    return torch.stack(losses).mean()


def flatten_probas(probas: torch.Tensor, labels: torch.Tensor, ignore_index: int = 255) -> tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = probas.shape
    probas = probas.transpose(1, 2).transpose(2, 3).contiguous().view(-1, C)
    labels = labels.view(-1)
    valid = labels != ignore_index
    return probas[valid], labels[valid]


def lovasz_softmax(probas: torch.Tensor, labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    probas_flat, labels_flat = flatten_probas(probas, labels, ignore_index)
    classes = list(range(probas.size(1)))
    return lovasz_softmax_flat(probas_flat, labels_flat, classes)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        class_weights: list[float] | None = None,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.register_buffer("class_weights", torch.ones(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, weight=self.class_weights, ignore_index=self.ignore_index
        )
        probas = F.softmax(logits.float(), dim=1)
        lov = lovasz_softmax(probas.float(), targets, self.ignore_index)
        total = self.ce_weight * ce + self.lovasz_weight * lov
        if torch.isnan(total) or torch.isinf(total):
            return ce
        return total
