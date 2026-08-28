#!/usr/bin/env python3
import argparse
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
from src.data.projection import build_range_image, build_label_image
from src.data.augmentation import RangeImageAugment


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


def compute_miou(preds: np.ndarray, targets: np.ndarray, num_classes: int = 4, ignore_index: int = 255) -> tuple[float, list[float]]:
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    class_ious = []
    for c in range(num_classes):
        pred_c = preds == c
        tgt_c = targets == c
        inter = np.sum(pred_c & tgt_c)
        uni = np.sum(pred_c | tgt_c)
        class_ious.append(float(inter / uni) if uni > 0 else float("nan"))
    valid_ious = [x for x in class_ious if not np.isnan(x)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return mean_iou, class_ious


def build_batch(points_list, labels_list, proj_list, device, augment, is_train, h, w, max_range):
    range_imgs = []
    label_imgs = []
    for i in range(len(points_list)):
        pts = points_list[i].cpu().numpy()
        lbl = labels_list[i].cpu().numpy()
        prj = proj_list[i].cpu().numpy()

        if is_train and augment is not None:
            ri, li, _ = augment(pts, prj, lbl)
        else:
            remissions = pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts))
            ranges = np.sqrt(np.sum(pts[:, :3] ** 2, axis=1))
            ri = build_range_image(pts, prj, remissions, ranges, h=h, w=w, max_range=max_range)
            li = build_label_image(lbl, prj, ranges, h=h, w=w)

        range_imgs.append(torch.from_numpy(ri))
        label_imgs.append(torch.from_numpy(li))

    range_img_batch = torch.stack(range_imgs).to(device, non_blocking=True)
    label_img_batch = torch.stack(label_imgs).to(device, non_blocking=True)
    return range_img_batch, label_img_batch


@torch.no_grad()
def validate(model, val_loader, loss_fn, device, cfg, augment, max_batches=50):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n = 0

    h, w = cfg["model"]["h"], cfg["model"]["w"]
    max_range = cfg["train"]["max_range"]

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        points, labels, proj = batch
        range_img_batch, label_img_batch = build_batch(
            points, labels, proj, device, None, False, h, w, max_range,
        )

        with torch.amp.autocast("cuda", enabled=True):
            logits = model(range_img_batch)
            loss = loss_fn(logits, label_img_batch)

        total_loss += loss.item()
        n += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        tgts = label_img_batch.cpu().numpy()
        all_preds.append(preds.reshape(-1))
        all_targets.append(tgts.reshape(-1))

    avg_loss = total_loss / max(n, 1)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    miou, class_ious = compute_miou(all_preds, all_targets)

    model.train()
    return avg_loss, miou, class_ious


def train(cfg: dict, resume: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed_root = cfg["data"]["processed_root"]
    meta_path = Path(processed_root) / "meta.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run scripts/preprocess_to_shards.py first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    class_weights = meta.get("class_weights", [1.0, 1.0, 1.0, 1.0])
    print(f"Class weights: {class_weights}")

    h, w = cfg["model"]["h"], cfg["model"]["w"]
    max_range = cfg["train"]["max_range"]

    model = RangeImageUNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        use_groupnorm=cfg["model"].get("use_groupnorm", True),
        groups=cfg["model"].get("groups", 8),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    loss_fn = CombinedLoss(
        num_classes=cfg["model"]["num_classes"],
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

    augment = RangeImageAugment(
        cfg["train"].get("augmentation", {}),
        h=h, w=w,
        fov_top_deg=cfg["model"]["fov_top_deg"],
        fov_bottom_deg=cfg["model"]["fov_bottom_deg"],
        max_range=max_range,
    )

    ckpt_dir = Path(cfg["checkpoint"]["dir"])
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
        model.load_state_dict(ckpt["model_state_dict"])
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"  Loaded best_miou={best_miou:.4f} (fresh training)")

    train_loader = create_train_loader(processed_root, batch_size=batch_size, num_workers=2)
    val_loader = create_val_loader(processed_root, batch_size=batch_size, num_workers=2)

    approx_steps_per_epoch = meta["n_train"] // batch_size
    total_steps = epochs * approx_steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    print(f"\nTraining: {epochs} epochs, ~{approx_steps_per_epoch} steps/epoch, val every {val_every} steps")
    print(f"Effective batch size: {batch_size * grad_accum}")
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
            points, labels, proj = batch

            range_img_batch, label_img_batch = build_batch(
                points, labels, proj, device, augment, True, h, w, max_range,
            )

            with torch.amp.autocast("cuda", enabled=True):
                logits = model(range_img_batch)
                loss = loss_fn(logits, label_img_batch)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
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
                val_loss, val_miou, class_ious = validate(
                    model, val_loader, loss_fn, device, cfg, augment,
                )
                lr_now = optimizer.param_groups[0]["lr"]

                print(f"  [Step {global_step}] val_loss={val_loss:.4f} val_miou={val_miou:.4f} lr={lr_now:.6f}")
                class_names = ["drivable", "terrain", "static", "dynamic"]
                for ci, cn in zip(class_ious, class_names):
                    print(f"    {cn}: {ci:.4f}")

                history_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_miou": val_miou,
                    "class_ious": dict(zip(class_names, class_ious)),
                }
                with open(history_path, "a") as f:
                    f.write(json.dumps(history_entry) + "\n")

                if val_miou > best_miou:
                    best_miou = val_miou
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
                    print(f"  *** New best miou: {best_miou:.4f} -> {best_path}")
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

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")


def main():
    import yaml
    parser = argparse.ArgumentParser(description="Train range-image UNet")
    parser.add_argument("--config", type=str, default="train_range_image")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent / "config" / f"{args.config}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
