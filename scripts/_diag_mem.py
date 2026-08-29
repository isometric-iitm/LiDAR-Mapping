import ctypes
import sys
import time
from ctypes import wintypes

sys.path.insert(0, ".")

psapi = ctypes.WinDLL("psapi.dll")


def rss_mb():
    class C(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    c = C()
    c.cb = ctypes.sizeof(C)
    psapi.GetProcessMemoryInfo(ctypes.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1e6, c.PagefileUsage / 1e6


import gc

import numpy as np
import torch
import torch.nn as nn

from src.data.webdataset_loader import create_train_loader, create_val_loader
from src.data.augmentation import RangeImageAugment
from src.data.label_mapping import bin_5_to_4
from src.models.unet import RangeImageUNet
from src.models.lovasz import CombinedLoss
from scripts.train import validate

device = torch.device("cuda")
model = RangeImageUNet(in_channels=5, num_classes=5, base_channels=32).to(device)
loss_fn = CombinedLoss(num_classes=5, class_weights=[1.0, 0.64, 1.0, 2.0, 5.0], lovasz_weight=0.5).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler = torch.amp.GradScaler("cuda")
aug = RangeImageAugment({})

train_loader = create_train_loader("F:/sih/processed", batch_size=2, num_workers=2)
val_loader = create_val_loader("F:/sih/processed", batch_size=2, num_workers=0)


def mark(name):
    ws, pf = rss_mb()
    print(f"  [{name:>24}] working_set={ws:8.1f}MB pagefile={pf:8.1f}MB")


mark("start")
it = iter(train_loader)
n_steps = 0
gc.disable()
while n_steps < 60:
    ri, li = next(it)
    ri = ri.to(device, non_blocking=True)
    li = li.to(device, non_blocking=True)
    ri, li = aug.batch_augment(ri, li)
    with torch.amp.autocast("cuda", enabled=True):
        logits = model(ri)
        loss = loss_fn(logits, li) / 4
    scaler.scale(loss).backward()
    if (n_steps + 1) % 4 == 0:
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    n_steps += 1
mark("60 train steps")
torch.cuda.empty_cache()
gc.collect()
mark("post empty_cache")

print("--- entering validation ---")
vl, m5, m4, c5, c4 = validate(model, val_loader, loss_fn, device, num_classes=5, max_batches=20)
mark("after validate")
gc.collect()
torch.cuda.empty_cache()
mark("validate + cleanup")
print(f"val_miou_4={m4:.4f}")