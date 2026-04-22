#!/usr/bin/env python3
"""
NetLexicon pretraining entry: dual-head NTP + VQ + contrastive learning.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NetLexiconConfig
from utils import set_seed, load_json, get_cosine_schedule_with_warmup
from model import NetLexiconPretrainModel
from pretrain.dataset_loader import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--type_loss_weight", type=float, default=None,
                        help="Weight for type prediction loss (ablation: 0 to disable).")
    parser.add_argument("--feat_loss_weight", type=float, default=None,
                        help="Weight for feature prediction loss (ablation: 0 to disable).")
    parser.add_argument("--flow_loss_weight", type=float, default=None,
                        help="Weight for contrastive flow loss (ablation: 0 to disable).")
    parser.add_argument("--output_ckpt", type=str, default=None,
                        help="Path to save best checkpoint (default: checkpoints/pretrain/best.pt).")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run name. Checkpoints/logs go to checkpoints/pretrain/<run_name>/ "
                             "and logs/pretrain/<run_name>/. With --ablation, goes to ablation/ subdirectory instead.")
    parser.add_argument("--ablation", action="store_true",
                        help="Save to checkpoints/ablation/<run_name>/pretrain/ instead of pretrain/<run_name>/.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. 'cuda:0' or 'cpu'. Default: auto-detect.")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Directory containing splits.json and label_to_idx.json "
                             "(e.g. 'splits/CipherSpectrum'). Default: project root.")
    # Model architecture overrides.
    parser.add_argument(
        "--legacy-heavy",
        action="store_true",
        help="Use legacy heavy config (e.g., 512-d/6 layers).",
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--vq_dim", type=int, default=None)
    parser.add_argument("--vq_codebook_size", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--contrastive_proj_dim", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def validate(model, val_loader, config, device):
    model.eval()
    total_loss = 0.0
    total_type = 0.0
    total_feat = 0.0
    total_flow = 0.0
    total_vq = 0.0
    total_ppl = 0.0
    n_batches = 0

    for batch in val_loader:
        (features, tt, vmask,
         type_target, feat_target, feat_mask,
         win_stats, win_mask, win_counts, labels) = batch
        features = features.to(device)
        tt = tt.to(device)
        vmask = vmask.to(device)
        type_target = type_target.to(device)
        feat_target = feat_target.to(device)
        feat_mask = feat_mask.to(device)
        win_stats = win_stats.to(device)
        win_mask = win_mask.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=config.use_amp):
            type_logits, feat_pred, vq_loss, perplexity, z_q_wins = model(
                features, tt, vmask)
            loss, type_loss, feat_loss, flow_loss = model.compute_loss(
                type_logits, feat_pred, type_target, feat_target, feat_mask,
                vq_loss, z_q_wins, win_stats, win_mask)

        total_loss += loss.item()
        total_type += type_loss.item()
        total_feat += feat_loss.item()
        total_flow += flow_loss.item()
        total_vq += vq_loss.item()
        total_ppl += perplexity.item()
        n_batches += 1

    n = max(1, n_batches)
    return (total_loss / n, total_type / n, total_feat / n,
            total_flow / n, total_vq / n, total_ppl / n)


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
    }, path)


def main():
    args = parse_args()
    config = NetLexiconConfig()

    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_amp:
        config.use_amp = False
    if args.type_loss_weight is not None:
        config.type_loss_weight = args.type_loss_weight
    if args.feat_loss_weight is not None:
        config.feat_loss_weight = args.feat_loss_weight
    if args.flow_loss_weight is not None:
        config.flow_loss_weight = args.flow_loss_weight
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
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.contrastive_proj_dim is not None:
        config.contrastive_proj_dim = args.contrastive_proj_dim

    set_seed(config.split_seed)
    if args.device is not None:
        device = torch.device(args.device)
        if str(device).startswith("cpu"):
            config.use_amp = False
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
    print("Building dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(config, norm_stats, label_to_idx, splits=splits)
    print(f"  train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    model = NetLexiconPretrainModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Num parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(enabled=config.use_amp)

    # Checkpoint & log directory
    if args.output_ckpt:
        best_ckpt_path = args.output_ckpt
    else:
        if args.ablation and args.run_name:
            best_ckpt_path = str(root / config.checkpoint_dir / "ablation" / args.run_name / "pretrain" / "best.pt")
        elif args.run_name:
            best_ckpt_path = str(root / config.checkpoint_dir / "pretrain" / args.run_name / "best.pt")
        else:
            best_ckpt_path = str(root / config.checkpoint_dir / "pretrain" / "best.pt")
    ckpt_dir = Path(best_ckpt_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation and args.run_name:
        log_dir = root / config.log_dir / "ablation" / args.run_name / "pretrain"
    elif args.run_name:
        log_dir = root / config.log_dir / "pretrain" / args.run_name
    else:
        log_dir = root / config.log_dir / "pretrain"
    writer = SummaryWriter(str(log_dir)) if SummaryWriter is not None else None

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_type = 0.0
        epoch_feat = 0.0
        epoch_flow = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.max_epochs - 1}",
                    dynamic_ncols=True, leave=True)
        for step, batch in enumerate(pbar):
            (features, tt, vmask,
             type_target, feat_target, feat_mask,
             win_stats, win_mask, win_counts, labels) = batch
            features = features.to(device)
            tt = tt.to(device)
            vmask = vmask.to(device)
            type_target = type_target.to(device)
            feat_target = feat_target.to(device)
            feat_mask = feat_mask.to(device)
            win_stats = win_stats.to(device)
            win_mask = win_mask.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=config.use_amp):
                type_logits, feat_pred, vq_loss, perplexity, z_q_wins = model(
                    features, tt, vmask)
                total_loss, type_loss, feat_loss, flow_loss = model.compute_loss(
                    type_logits, feat_pred, type_target, feat_target, feat_mask,
                    vq_loss, z_q_wins, win_stats, win_mask)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            epoch_loss += total_loss.item()
            epoch_type += type_loss.item()
            epoch_feat += feat_loss.item()
            epoch_flow += flow_loss.item()

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                type=f"{type_loss.item():.4f}",
                feat=f"{feat_loss.item():.4f}",
                ppl=f"{perplexity.item():.1f}",
            )

            if global_step % config.log_every_n_steps == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  [E{epoch} S{step + 1}] loss={total_loss.item():.4f} "
                      f"type={type_loss.item():.4f} feat={feat_loss.item():.4f} "
                      f"flow={flow_loss.item():.4f} vq={vq_loss.item():.4f} "
                      f"ppl={perplexity.item():.1f} lr={lr:.2e}")
                if writer is not None:
                    writer.add_scalar("train/total_loss", total_loss.item(), global_step)
                    writer.add_scalar("train/type_loss", type_loss.item(), global_step)
                    writer.add_scalar("train/feat_loss", feat_loss.item(), global_step)
                    writer.add_scalar("train/flow_loss", flow_loss.item(), global_step)
                    writer.add_scalar("train/vq_loss", vq_loss.item(), global_step)
                    writer.add_scalar("train/perplexity", perplexity.item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)

        n_steps = len(train_loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}: avg_loss={epoch_loss / n_steps:.4f} "
              f"avg_type={epoch_type / n_steps:.4f} "
              f"avg_feat={epoch_feat / n_steps:.4f} "
              f"avg_flow={epoch_flow / n_steps:.4f} time={elapsed:.0f}s")

        del features, tt, vmask, type_target, feat_target, feat_mask
        del win_stats, win_mask, win_counts, labels
        del type_logits, feat_pred, vq_loss, perplexity, z_q_wins
        del total_loss, type_loss, feat_loss, flow_loss
        torch.cuda.empty_cache()

        if (epoch + 1) % config.val_every_n_epochs == 0:
            val_results = validate(model, val_loader, config, device)
            val_loss, val_type, val_feat, val_flow, val_vq, val_ppl = val_results
            torch.cuda.empty_cache()
            print(f"  Val: loss={val_loss:.4f} type={val_type:.4f} "
                  f"feat={val_feat:.4f} flow={val_flow:.4f} "
                  f"vq={val_vq:.4f} ppl={val_ppl:.1f}")
            if writer is not None:
                writer.add_scalar("val/total_loss", val_loss, epoch)
                writer.add_scalar("val/type_loss", val_type, epoch)
                writer.add_scalar("val/feat_loss", val_feat, epoch)
                writer.add_scalar("val/flow_loss", val_flow, epoch)
                writer.add_scalar("val/vq_loss", val_vq, epoch)
                writer.add_scalar("val/perplexity", val_ppl, epoch)

            # Save epoch checkpoint (next to best).
            epoch_ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                            str(epoch_ckpt_path))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_ckpt_path)
                print(f"  ** saved best checkpoint (val_loss={val_loss:.4f}) -> {best_ckpt_path}")

    if writer is not None:
        writer.close()
    print(f"\nTraining done. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
