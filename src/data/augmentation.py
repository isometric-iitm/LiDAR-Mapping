import torch
import numpy as np


class RangeImageAugment:
    def __init__(self, cfg: dict):
        self.flip_h = cfg.get("random_flip_h", True)
        self.flip_h_prob = cfg.get("random_flip_h_prob", 0.5)
        self.intensity_jitter = cfg.get("intensity_jitter", True)
        self.intensity_jitter_prob = cfg.get("intensity_jitter_prob", 0.3)
        self.intensity_jitter_std = cfg.get("intensity_jitter_std", 0.05)
        self.cutout = cfg.get("cutout", True)
        self.cutout_prob = cfg.get("cutout_prob", 0.3)
        self.cutout_h = cfg.get("cutout_h", 16)
        self.cutout_w = cfg.get("cutout_w", 256)

    def __call__(self, ri: torch.Tensor, li: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flip_h and torch.rand(1).item() < self.flip_h_prob:
            ri = torch.flip(ri, [2])
            li = torch.flip(li, [1])

        if self.intensity_jitter and torch.rand(1).item() < self.intensity_jitter_prob:
            jitter = float(torch.normal(1.0, self.intensity_jitter_std, (1,)).item())
            ri[4] = (ri[4] * jitter).clamp(0.0, 1.0)

        if self.cutout and torch.rand(1).item() < self.cutout_prob:
            h, w = ri.shape[1], ri.shape[2]
            ch = int(torch.randint(0, max(h - self.cutout_h, 1), (1,)).item())
            cw = int(torch.randint(0, max(w - self.cutout_w, 1), (1,)).item())
            ri[:, ch:ch + self.cutout_h, cw:cw + self.cutout_w] = 0.0

        return ri, li

    def batch_augment(self, ri_batch: torch.Tensor, li_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = ri_batch.shape[0]
        for b in range(B):
            ri_batch[b], li_batch[b] = self(ri_batch[b], li_batch[b])
        return ri_batch, li_batch
