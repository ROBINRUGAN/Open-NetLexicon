#!/usr/bin/env python3
"""
Evaluation script: compute metrics on the test set.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NetLexiconConfig
from utils import set_seed, load_json, torch_load_checkpoint
from model import NetLexiconPretrainModel, NetLexiconFinetuneModel
from pretrain.dataset_loader import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_per_class", type=str, default=None,
                        help="Save per-class P/R/F1 list to JSON (finetune mode only).")
    parser.add_argument("--metrics_json", type=str, default=None,
                        help="Save overall test metrics (accuracy, macro_f1, loss, ...) to JSON (finetune).")
    parser.add_argument("--per_class_csv", type=str, default=None,
                        help="Save per-class breakdown to CSV (finetune).")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Directory containing splits.json and label_to_idx.json "
                             "(e.g. 'splits/CipherSpectrum'). Default: project root.")
    # Model architecture overrides (must match checkpoint).
    parser.add_argument(
        "--legacy-heavy",
        action="store_true",
        help="Use legacy heavy config (e.g., 512-d/6 layers) and match legacy checkpoints.",
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--vq_dim", type=int, default=None)
    parser.add_argument("--vq_codebook_size", type=int, default=None)
    parser.add_argument("--contrastive_proj_dim", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_pretrain(model, test_loader, config, device):
    model.eval()
    total_type = 0.0
    total_feat = 0.0
    total_vq = 0.0
    total_ppl = 0.0
    n = 0
    for batch in test_loader:
        (features, tt, vmask,
         type_target, feat_target, feat_mask,
         win_stats, win_mask, win_counts, _labels) = batch
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
            _total, type_loss, feat_loss, flow_loss = model.compute_loss(
                type_logits, feat_pred, type_target, feat_target, feat_mask,
                vq_loss, z_q_wins, win_stats, win_mask)

        total_type += type_loss.item()
        total_feat += feat_loss.item()
        total_vq += vq_loss.item()
        total_ppl += perplexity.item()
        n += 1

    print("\n=== Pretrain eval (test set) ===")
    print(f"  Type loss:  {total_type / n:.6f}")
    print(f"  Feat loss:  {total_feat / n:.6f}")
    print(f"  VQ loss:    {total_vq / n:.6f}")
    print(f"  Perplexity: {total_ppl / n:.1f}")


@torch.no_grad()
def evaluate_finetune(model, test_loader, config, device, idx_to_label, quiet=False, show_progress=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    batch_iter = tqdm(test_loader, desc="Test", leave=False, disable=not show_progress)
    for batch in batch_iter:
        (features, tt, vmask,
         _type_target, _feat_target, _feat_mask,
         _win_stats, _win_mask, _win_counts, labels) = batch
        features = features.to(device)
        tt = tt.to(device)
        vmask = vmask.to(device)
        labels = labels.to(device)

        logits, _ = model(features, tt, vmask)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n_batches = max(1, len(test_loader))
    n_samples = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / max(1, n_samples)

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    for p, l in zip(all_preds, all_labels):
        if p == l:
            per_class_tp[l] += 1
        else:
            per_class_fp[p] += 1
            per_class_fn[l] += 1

    classes = sorted(set(all_labels))
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = per_class_tp[c]
        fp = per_class_fp[c]
        fn = per_class_fn[c]
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-8, p + r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_p = sum(precisions) / max(1, len(precisions))
    macro_r = sum(recalls) / max(1, len(recalls))
    macro_f1 = sum(f1s) / max(1, len(f1s))

    per_class = []
    for c in classes:
        label_name = idx_to_label.get(c, str(c))
        tp = per_class_tp[c]
        fp = per_class_fp[c]
        fn = per_class_fn[c]
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-8, p + r)
        per_class.append({"class": label_name, "precision": round(p, 4), "recall": round(r, 4), "f1-score": round(f1, 4)})

    test_loss_avg = total_loss / n_batches
    metrics = {
        "accuracy": accuracy,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "test_loss": test_loss_avg,
        "n_samples": n_samples,
        "per_class": per_class,
    }

    if not quiet:
        print(f"\n=== Finetune eval (test set, n={n_samples}) ===")
        print(f"  Loss:       {test_loss_avg:.4f}")
        print(f"  Accuracy:   {accuracy:.4f} ({correct}/{n_samples})")
        print(f"  Macro P:    {macro_p:.4f}")
        print(f"  Macro R:    {macro_r:.4f}")
        print(f"  Macro F1:   {macro_f1:.4f}")
        print(f"\n  Per-class F1:")
        for c in classes:
            label_name = idx_to_label.get(c, str(c))
            tp = per_class_tp[c]
            fp = per_class_fp[c]
            fn = per_class_fn[c]
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1 = 2 * p * r / max(1e-8, p + r)
            print(f"    [{c:2d}] {label_name:30s}  P={p:.3f} R={r:.3f} F1={f1:.3f}")

    return metrics


def run_finetune_eval_return_metrics(checkpoint_path, config=None, quiet=True, show_progress=False, splits_dir=None):
    """Load finetune checkpoint, run on test set, return dict with accuracy, macro_f1, etc."""
    if config is None:
        config = NetLexiconConfig()
    config.num_workers = getattr(config, "num_workers", 2)
    set_seed(config.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(config.project_root)
    norm_stats = load_json(root / config.norm_stats_path)
    splits_base = (root / splits_dir) if splits_dir else root
    label_to_idx = load_json(splits_base / config.label_to_idx_path)
    splits = load_json(splits_base / config.splits_path)
    config.num_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    _, _, test_loader = build_dataloaders(config, norm_stats, label_to_idx, splits=splits)
    pretrained = NetLexiconPretrainModel(config)
    ckpt = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    use_vq = ckpt.get("use_vq", True)
    model = NetLexiconFinetuneModel(config, pretrained, use_vq=use_vq)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    return evaluate_finetune(model, test_loader, config, device, idx_to_label, quiet=quiet, show_progress=show_progress)


def main():
    args = parse_args()
    config = NetLexiconConfig()
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
    set_seed(config.split_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(config.project_root)
    norm_stats = load_json(root / config.norm_stats_path)

    if args.splits_dir:
        splits_base = root / args.splits_dir
    else:
        splits_base = root
    label_to_idx = load_json(splits_base / config.label_to_idx_path)
    splits = load_json(splits_base / config.splits_path)
    config.num_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    print(f"Splits: {splits_base}")
    print(f"Num classes: {config.num_classes}")
    print("Building test dataloader...")
    _, _, test_loader = build_dataloaders(config, norm_stats, label_to_idx, splits=splits)
    print(f"  test batches: {len(test_loader)}")

    if args.mode == "pretrain":
        model = NetLexiconPretrainModel(config)
        ckpt = torch_load_checkpoint(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        evaluate_pretrain(model, test_loader, config, device)

    elif args.mode == "finetune":
        pretrained = NetLexiconPretrainModel(config)
        ckpt = torch_load_checkpoint(args.checkpoint, map_location="cpu")
        use_vq = ckpt.get("use_vq", True)
        model = NetLexiconFinetuneModel(config, pretrained, use_vq=use_vq)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        metrics = evaluate_finetune(model, test_loader, config, device, idx_to_label)
        if args.save_per_class and "per_class" in metrics:
            out_path = Path(args.save_per_class)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics["per_class"], f, indent=2, ensure_ascii=False)
            print(f"  Saved per-class metrics -> {out_path}")
        if args.metrics_json:
            out_path = Path(args.metrics_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "accuracy": metrics["accuracy"],
                "macro_p": metrics["macro_p"],
                "macro_r": metrics["macro_r"],
                "macro_f1": metrics["macro_f1"],
                "test_loss": metrics["test_loss"],
                "n_samples": metrics["n_samples"],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  Saved test metrics -> {out_path}")
        if args.per_class_csv and "per_class" in metrics:
            out_path = Path(args.per_class_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["class", "precision", "recall", "f1-score"])
                for row in metrics["per_class"]:
                    w.writerow(
                        [
                            row["class"],
                            row["precision"],
                            row["recall"],
                            row["f1-score"],
                        ]
                    )
            print(f"  Saved per-class CSV -> {out_path}")


if __name__ == "__main__":
    main()
