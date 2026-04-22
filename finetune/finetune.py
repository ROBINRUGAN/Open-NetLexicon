#!/usr/bin/env python3
"""
NetLexicon fine-tuning entry (classification).

Schedule:
  1) First freeze_encoder_epochs: freeze embedding/encoder/vq and train classifier only
  2) Then unfreeze and fine-tune end-to-end with a smaller LR
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NetLexiconConfig
from utils import set_seed, load_json, get_cosine_schedule_with_warmup, torch_load_checkpoint
from model import NetLexiconPretrainModel, NetLexiconFinetuneModel
from finetune.dataset_loader import build_finetune_dataloaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint. Omit when using --no_pretrained.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override config.finetune_batch_size (for consistent benchmarking).")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--frozen_only", action="store_true",
                        help="Freeze encoder for all epochs (only train classifier).")
    parser.add_argument("--no_vq", action="store_true",
                        help="Use encoder hidden at [WIN] instead of z_q (ablation: no VQ).")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Start from random init (ablation: no pretraining).")
    parser.add_argument("--output_ckpt", type=str, default=None,
                        help="Path to save best checkpoint (default: checkpoints/finetune/best.pt).")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run name. Checkpoints/logs go to checkpoints/finetune/<run_name>/ "
                             "and logs/finetune/<run_name>/. With --ablation, goes to ablation/ subdirectory instead.")
    parser.add_argument("--ablation", action="store_true",
                        help="Save to checkpoints/ablation/<run_name>/ instead of finetune/<run_name>/.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. 'cuda:0' or 'cpu'. Default: auto-detect.")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Directory containing splits.json and label_to_idx.json "
                             "(e.g. 'splits/CipherSpectrum'). Default: project root.")
    # Model architecture overrides (must match the pretrained checkpoint).
    parser.add_argument(
        "--legacy-heavy",
        action="store_true",
        help="Use legacy heavy config (e.g., 512-d/6 layers). Must match legacy pretrained weights.",
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--vq_dim", type=int, default=None)
    parser.add_argument("--vq_codebook_size", type=int, default=None)
    parser.add_argument("--contrastive_proj_dim", type=int, default=None)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="If >0, exit after this many steps (memory probing). 0 means normal training.",
    )
    parser.add_argument(
        "--step_bench_warmup",
        type=int,
        default=10,
        help="Drop first N warmup steps for step-time benchmarking (only used with --max_train_steps).",
    )
    parser.add_argument(
        "--freeze_encoder_epochs",
        type=int,
        default=None,
        help="Override config.freeze_encoder_epochs (use 0 for immediate end-to-end backprop when probing).",
    )
    parser.add_argument(
        "--bench_first_epoch_only",
        action="store_true",
        help="Run only epoch 1, print [EPOCH_BENCH] timings (train/val), then exit.",
    )
    return parser.parse_args()


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in val_loader:
        (features, tt, vmask,
         _type_target, _feat_target, _feat_mask,
         _win_stats, _win_mask, _win_counts, labels) = batch
        features = features.to(device)
        tt = tt.to(device)
        vmask = vmask.to(device)
        labels = labels.to(device)
        logits, vq_loss = model(features, tt, vmask)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    n = max(1, len(val_loader))
    return total_loss / n, correct / max(1, total)


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, path, use_vq=True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "use_vq": use_vq,
    }, path)


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def main():
    args = parse_args()
    config = NetLexiconConfig()

    if args.epochs is not None:
        config.finetune_epochs = args.epochs
    if args.batch_size is not None:
        config.finetune_batch_size = args.batch_size
    if args.lr is not None:
        config.finetune_lr = args.lr
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.legacy_heavy:
        config.d_model = 512
        config.n_layers = 6
        config.n_heads = 8
        config.d_ff = 2048
        config.vq_dim = 512
        config.vq_codebook_size = 512
        config.contrastive_proj_dim = 128
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.n_heads is not None:
        config.n_heads = args.n_heads
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.vq_dim is not None:
        config.vq_dim = args.vq_dim
    if args.vq_codebook_size is not None:
        config.vq_codebook_size = args.vq_codebook_size
    if args.contrastive_proj_dim is not None:
        config.contrastive_proj_dim = args.contrastive_proj_dim

    if not args.no_pretrained and not args.pretrained:
        raise SystemExit("Error: --pretrained is required unless --no_pretrained is set.")

    if args.frozen_only:
        config.freeze_encoder_epochs = config.finetune_epochs
        print("Ablation: frozen_only (encoder frozen for all epochs).")
    if args.freeze_encoder_epochs is not None:
        config.freeze_encoder_epochs = args.freeze_encoder_epochs

    use_vq = not args.no_vq
    if args.no_vq:
        print("Ablation: no_vq (use encoder hidden at [WIN] instead of z_q).")

    set_seed(config.split_seed)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root = Path(config.project_root)
    norm_stats = load_json(root / config.norm_stats_path)

    if args.splits_dir:
        splits_base = root / args.splits_dir
    else:
        splits_base = root
    label_to_idx = load_json(splits_base / config.label_to_idx_path)
    splits = load_json(splits_base / config.splits_path)
    config.num_classes = len(label_to_idx)

    print(f"Splits: {splits_base}")
    print(f"Num classes: {config.num_classes}")

    pretrained = NetLexiconPretrainModel(config)
    if args.no_pretrained:
        print("Ablation: no_pretrained (random init).")
    else:
        print("Loading pretrained checkpoint...")
        ckpt = torch_load_checkpoint(args.pretrained, map_location="cpu")
        pretrained.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded: {args.pretrained} (epoch {ckpt.get('epoch', '?')})")

    model = NetLexiconFinetuneModel(config, pretrained, use_vq=use_vq).to(device)
    del pretrained

    print("Building dataloaders...")
    train_loader, val_loader, _ = build_finetune_dataloaders(
        config, norm_stats, label_to_idx, splits=splits
    )
    print(f"  train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    criterion = nn.CrossEntropyLoss()

    # Checkpoint & log directory
    if args.output_ckpt:
        save_path = args.output_ckpt
    elif args.run_name:
        if args.ablation:
            save_path = str(root / config.checkpoint_dir / "ablation" / args.run_name / "finetune" / "best.pt")
        else:
            save_path = str(root / config.checkpoint_dir / "finetune" / args.run_name / "best.pt")
    else:
        save_path = str(root / config.checkpoint_dir / "finetune" / "best.pt")
    ckpt_dir = Path(save_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.run_name:
        if args.ablation:
            log_dir = root / config.log_dir / "ablation" / args.run_name / "finetune"
        else:
            log_dir = root / config.log_dir / "finetune" / args.run_name
    else:
        log_dir = root / config.log_dir / "finetune"
    writer = SummaryWriter(str(log_dir)) if SummaryWriter is not None else None

    best_val_acc = 0.0
    global_step = 0
    if device.type == "cuda" and getattr(args, "max_train_steps", 0):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for epoch in range(config.finetune_epochs):
        if epoch < config.freeze_encoder_epochs:
            freeze_backbone(model)
            lr = config.finetune_lr * 10
        else:
            unfreeze_all(model)
            lr = config.finetune_lr

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=config.weight_decay)

        model.train()
        epoch_loss = 0.0
        correct = 0
        total_samples = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.finetune_epochs - 1}",
                    dynamic_ncols=True, leave=True)
        mem_stop = False
        step_times_ms: list[float] = []
        for step, batch in enumerate(pbar):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _t_step_begin = time.perf_counter()
            (features, tt, vmask,
             _type_target, _feat_target, _feat_mask,
             _win_stats, _win_mask, _win_counts, labels) = batch
            features = features.to(device)
            tt = tt.to(device)
            vmask = vmask.to(device)
            labels = labels.to(device)

            logits, vq_loss = model(features, tt, vmask)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times_ms.append((time.perf_counter() - _t_step_begin) * 1000.0)

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct / max(1, total_samples):.4f}")

            global_step += 1
            if args.max_train_steps and global_step >= args.max_train_steps:
                mem_stop = True
                break

        if mem_stop:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                pa = torch.cuda.max_memory_allocated(device) / (1024**3)
                pr = torch.cuda.max_memory_reserved(device) / (1024**3)
                print(
                    f"[MEM_BENCH] peak_allocated_gb={pa:.4f} peak_reserved_gb={pr:.4f}"
                )
            _warm = getattr(args, "step_bench_warmup", 10)
            measured = step_times_ms[_warm:]
            if measured:
                import statistics as _st
                _mean = sum(measured) / len(measured)
                _std = _st.pstdev(measured) if len(measured) > 1 else 0.0
                print(
                    f"[STEP_BENCH] step_time_ms_mean={_mean:.4f} step_time_ms_std={_std:.4f} "
                    f"n_measured={len(measured)} n_warmup={_warm} batch_size={config.finetune_batch_size}"
                )
            print(f"[MEM_BENCH] stopped after {global_step} train steps")
            break

        del features, tt, vmask, labels, logits, vq_loss, loss, preds
        torch.cuda.empty_cache()

        n_steps = len(train_loader)
        train_acc = correct / max(1, total_samples)
        train_wall_sec = time.time() - t0
        frozen_str = " [frozen]" if epoch < config.freeze_encoder_epochs else ""
        print(f"Epoch {epoch}{frozen_str}: loss={epoch_loss / n_steps:.4f} "
              f"acc={train_acc:.4f} time={train_wall_sec:.0f}s")

        val_t0 = time.perf_counter()
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_wall_sec = time.perf_counter() - val_t0
        torch.cuda.empty_cache()
        print(f"  Val: loss={val_loss:.4f} acc={val_acc:.4f}")
        if args.bench_first_epoch_only:
            print(
                f"[EPOCH_BENCH] epoch_train_sec={train_wall_sec:.4f} "
                f"epoch_val_sec={val_wall_sec:.4f} epoch_id={epoch}"
            )
        if writer is not None:
            writer.add_scalar("finetune/train_loss", epoch_loss / n_steps, epoch)
            writer.add_scalar("finetune/train_acc", train_acc, epoch)
            writer.add_scalar("finetune/val_loss", val_loss, epoch)
            writer.add_scalar("finetune/val_acc", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path, use_vq=use_vq)
            print(f"  ** saved best checkpoint (val_acc={val_acc:.4f}) -> {save_path}")

        if args.bench_first_epoch_only:
            print("[EPOCH_BENCH] bench_first_epoch_only: done (exiting after epoch 1).")
            break

    if writer is not None:
        writer.close()
    print(f"\nFine-tuning done. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
