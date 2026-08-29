#!/usr/bin/env python3
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.unet import RangeImageUNet
from src.models.lovasz import CombinedLoss
from src.data.webdataset_loader import create_train_loader, create_val_loader
from src.data.augmentation import RangeImageAugment
from src.data.label_mapping import bin_5_to_4


CLASS_NAMES_5 = ["drivable", "terrain_nondrivable", "static_obstacle", "dynamic_vehicle", "dynamic_pedestrian"]
CLASS_NAMES_4 = ["drivable", "terrain_nondrivable", "static_obstacle", "dynamic_object"]


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, step: int):
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale


def miou_from_confusion(cm: np.ndarray) -> tuple[float, list[float]]:
    num_classes = cm.shape[0]
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - inter
    class_ious = []
    for c in range(num_classes):
        class_ious.append(float(inter[c] / union[c]) if union[c] > 0 else float("nan"))
    valid_ious = [x for x in class_ious if not np.isnan(x)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return mean_iou, class_ious


def update_confusion(cm: np.ndarray, preds: np.ndarray, targets: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    valid = targets != ignore_index
    p = preds[valid].reshape(-1)
    t = targets[valid].reshape(-1)
    np.add.at(cm, (t, p), 1)
    return cm


@torch.no_grad()
def validate(model, val_loader, loss_fn, device, num_classes: int, max_batches: int = 20):
    model.eval()
    total_loss = 0.0
    cm_5 = np.zeros((num_classes, num_classes), dtype=np.int64)
    cm_4 = np.zeros((4, 4), dtype=np.int64)
    n = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        ri_batch, li_batch = batch
        ri_batch = ri_batch.to(device, non_blocking=True)
        li_batch = li_batch.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=True):
            logits = model(ri_batch)
            loss = loss_fn(logits, li_batch)

        total_loss += loss.item()
        n += 1

        preds_5 = logits.argmax(dim=1).cpu().numpy()
        tgts_5 = li_batch.cpu().numpy()
        preds_4 = bin_5_to_4(preds_5)
        tgts_4 = bin_5_to_4(tgts_5)

        update_confusion(cm_5, preds_5, tgts_5)
        update_confusion(cm_4, preds_4, tgts_4)

        del logits, preds_5, tgts_5, preds_4, tgts_4
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(n, 1)
    miou_5, class_ious_5 = miou_from_confusion(cm_5)
    miou_4, class_ious_4 = miou_from_confusion(cm_4)

    model.train()
    return avg_loss, miou_5, miou_4, class_ious_5, class_ious_4


def train(cfg: dict, resume: str | None = None,
          processed_root: str | Path | None = None,
          ckpt_dir: str | Path | None = None):
    from src.common.config import resolve_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed_root = resolve_path(processed_root or cfg["data"]["processed_root"])
    meta_path = Path(processed_root) / "meta.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run scripts/preprocess_to_shards.py first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("format") != "precomputed_rili":
        print(f"ERROR: Data format is '{meta.get('format')}'. Re-run preprocess_to_shards.py --force to generate precomputed ri/li shards.")
        return

    num_classes = cfg["model"]["num_classes"]
    class_weights = meta.get("class_weights", [1.0] * num_classes)
    if len(class_weights) != num_classes:
        print(f"WARNING: class_weights length {len(class_weights)} != num_classes {num_classes}, using ones")
        class_weights = [1.0] * num_classes
    print(f"Class weights: {class_weights}")

    model = RangeImageUNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=num_classes,
        base_channels=cfg["model"]["base_channels"],
        use_groupnorm=cfg["model"].get("use_groupnorm", True),
        groups=cfg["model"].get("groups", 8),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    loss_fn = CombinedLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        ce_weight=cfg["train"]["loss"]["ce_weight"],
        lovasz_weight=cfg["train"]["loss"]["lovasz_weight"],
        ignore_index=255,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    scaler = torch.amp.GradScaler("cuda")

    batch_size = cfg["train"]["batch_size"]
    grad_accum = cfg["train"]["grad_accum_steps"]
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"]["patience"]
    val_every = cfg["train"]["val_every_steps"]
    warmup_steps = cfg["train"]["warmup_steps"]
    grad_clip = cfg["train"]["grad_clip"]

    augment = RangeImageAugment(cfg["train"].get("augmentation", {}))

    ckpt_dir = resolve_path(ckpt_dir or cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / cfg["checkpoint"]["best_name"]
    last_path = ckpt_dir / cfg["checkpoint"]["last_name"]
    history_path = ckpt_dir / "history.jsonl"

    start_epoch = 0
    global_step = 0
    best_miou = 0.0
    patience_counter = 0

    if resume and Path(resume).exists():
        print(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"  epoch={start_epoch} step={global_step} best_miou={best_miou:.4f}")
    elif best_path.exists():
        print(f"Found existing best checkpoint at {best_path}")
        ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            best_miou = ckpt.get("best_miou", 0.0)
            print(f"  Loaded best_miou={best_miou:.4f} (fresh training)")
        except RuntimeError as e:
            print(f"  Checkpoint incompatible (likely num_classes changed): {e}")
            print(f"  Starting fresh training from scratch")

    num_workers = cfg["data"].get("num_workers", 0)
    train_loader = create_train_loader(processed_root, batch_size=batch_size, num_workers=num_workers)
    val_loader = create_val_loader(processed_root, batch_size=batch_size, num_workers=0)

    approx_steps_per_epoch = meta["n_train"] // batch_size
    total_steps = epochs * approx_steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    print(f"\nTraining: {epochs} epochs, ~{approx_steps_per_epoch} steps/epoch, val every {val_every} steps")
    print(f"Effective batch size: {batch_size * grad_accum}")
    print(f"Num classes: {num_classes} (train) -> 4 (binned eval)")
    print(f"Augmentation: enabled\n")

    log_every = 50
    running_loss = 0.0
    running_n = 0
    step_t0 = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            ri_batch, li_batch = batch
            ri_gpu = ri_batch.to(device, non_blocking=True)
            li_gpu = li_batch.to(device, non_blocking=True)

            ri_gpu, li_gpu = augment.batch_augment(ri_gpu, li_gpu)

            with torch.amp.autocast("cuda", enabled=True):
                logits = model(ri_gpu)
                loss = loss_fn(logits, li_gpu)
                loss = loss / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(global_step)

            epoch_loss += loss.item() * grad_accum
            epoch_steps += 1
            global_step += 1
            running_loss += loss.item() * grad_accum
            running_n += 1

            if global_step % log_every == 0 and global_step > 0:
                avg_rloss = running_loss / running_n
                dt = time.time() - step_t0
                steps_per_sec = running_n / max(dt, 0.001)
                lr_now = optimizer.param_groups[0]["lr"]
                vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
                print(f"  step {global_step:>6d} | loss {avg_rloss:.4f} | lr {lr_now:.2e} | {steps_per_sec:.1f} steps/s | vram {vram_mb:.0f}MB")
                running_loss = 0.0
                running_n = 0
                step_t0 = time.time()
                torch.cuda.reset_peak_memory_stats(device)

            if global_step % val_every == 0 and global_step > 0:
                val_loss, miou_5, miou_4, ci_5, ci_4 = validate(
                    model, val_loader, loss_fn, device, num_classes=num_classes,
                )
                lr_now = optimizer.param_groups[0]["lr"]
                gc.collect()
                torch.cuda.empty_cache()

                print(f"  [Step {global_step}] val_loss={val_loss:.4f} | miou_5class={miou_5:.4f} miou_4class={miou_4:.4f} | lr={lr_now:.6f}")
                for ci, cn in zip(ci_5, CLASS_NAMES_5):
                    v = f"{ci:.4f}" if not np.isnan(ci) else "N/A"
                    print(f"    5cls {cn}: {v}")
                for ci, cn in zip(ci_4, CLASS_NAMES_4):
                    v = f"{ci:.4f}" if not np.isnan(ci) else "N/A"
                    print(f"    4cls {cn}: {v}")

                history_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "miou_5class": miou_5,
                    "miou_4class": miou_4,
                    "class_ious_5": dict(zip(CLASS_NAMES_5, [float(x) if not np.isnan(x) else None for x in ci_5])),
                    "class_ious_4": dict(zip(CLASS_NAMES_4, [float(x) if not np.isnan(x) else None for x in ci_4])),
                }
                with open(history_path, "a") as f:
                    f.write(json.dumps(history_entry) + "\n")

                if miou_4 > best_miou:
                    best_miou = miou_4
                    patience_counter = 0
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "best_miou": best_miou,
                        "config": cfg,
                        "class_weights": class_weights,
                    }, str(best_path))
                    print(f"  *** New best miou_4class: {best_miou:.4f} -> {best_path}")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\nEarly stopping at step {global_step} (patience={patience})")
                    return

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"Epoch {epoch}: loss={avg_loss:.4f} time={elapsed:.1f}s vram={vram_mb:.0f}MB")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
            "best_miou": best_miou,
            "config": cfg,
            "class_weights": class_weights,
        }, str(last_path))

    print(f"\nTraining complete. Best miou_4class: {best_miou:.4f}")


def main():
    from src.common.config import load_config
    parser = argparse.ArgumentParser(description="Train range-image UNet")
    parser.add_argument("--config", type=str, default="train_range_image")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--processed-root", type=str, default=None,
                        help="Root with train/val shards + meta.json (default: config + PC2D_PROCESSED_ROOT)")
    parser.add_argument("--raw-root", type=str, default=None,
                        help="Root of raw sequences (default: config + PC2D_RAW_ROOT)")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Directory for checkpoints + history.jsonl (default: config + PC2D_CKPT_DIR)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI flags take highest priority; raw_root only matters if a later step needs it.
    if args.raw_root:
        cfg["data"]["raw_root"] = args.raw_root
    train(cfg, resume=args.resume,
          processed_root=args.processed_root,
          ckpt_dir=args.ckpt_dir)


if __name__ == "__main__":
    main()
