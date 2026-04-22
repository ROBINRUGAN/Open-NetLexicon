#!/usr/bin/env python3
"""
VQ codebook analysis: active code stats and correlations with window_stats / labels.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from itertools import islice

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NetLexiconConfig, TT_WIN
from model import NetLexiconPretrainModel
from tqdm import tqdm
from utils import load_json, set_seed, torch_load_checkpoint
from pretrain.dataset_loader import build_dataloaders
from analysis._codebook_utils import WS_NAMES, FLOW_MISSING  # shared window_stats metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze VQ codebook: active codes + stats vs window_stats/labels.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/pretrain/best.pt")
    parser.add_argument("--n_batches", type=int, default=200,
                        help="Number of val batches to run (default 200).")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Show top-k most used codes in detail (default 20).")
    parser.add_argument("--min_count", type=int, default=10,
                        help="Treat code as 'alive' if count >= this (default 10).")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Path relative to project_root (e.g. splits/CipherSpectrum).")
    parser.add_argument("--split", type=str, choices=("train", "val", "test"), default="val",
                        help="Which split to use (default: val; use test for final analysis).")
    parser.add_argument("--full", action="store_true",
                        help="Run the full split (ignore --n_batches).")
    parser.add_argument("--from_export", type=str, default=None,
                        help="Offline mode: load from an export directory and skip forward passes.")
    return parser.parse_args()


@torch.no_grad()
def collect_code_assignments(model, val_loader, n_batches, device, K):
    """Run model on val batches; for each [WIN] get code index, win_stats, label."""
    model.eval()
    code_to_stats = defaultdict(list)   # code_id -> list of (win_stats_37, label)
    code_counts = np.zeros(K, dtype=np.int64)

    n_actual = min(n_batches, len(val_loader))
    batch_iter = islice(val_loader, n_actual)
    for batch in tqdm(batch_iter, total=n_actual, desc="Codebook analysis", unit="batch"):
        (features, tt, vmask,
         _type_target, _feat_target, _feat_mask,
         win_stats, win_mask, _win_counts, labels) = batch
        features = features.to(device)
        tt = tt.to(device)
        vmask = vmask.to(device)
        win_stats = win_stats.to(device)
        win_mask = win_mask.to(device)
        labels = labels.to(device)

        h = model.embedding(features, tt)
        h = model.encoder(h, vmask)
        win_mask_flat = (tt == TT_WIN)  # [B, L]
        z_e = h[win_mask_flat]          # [N_win, D]

        if z_e.shape[0] == 0:
            continue

        codebook = model.vq.codebook.weight.float()
        dist = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = dist.argmin(dim=1).cpu().numpy()  # [N_win]

        flat_win_stats = win_stats[win_mask].cpu().numpy()  # [N_win, 37], same order as z_e
        win_labels = []
        for b in range(tt.shape[0]):
            for w in range(win_mask.shape[1]):
                if win_mask[b, w]:
                    win_labels.append(labels[b].item())
        win_labels = np.array(win_labels)

        for i, code_id in enumerate(indices):
            code_counts[code_id] += 1
            code_to_stats[code_id].append((flat_win_stats[i], win_labels[i]))

    return code_counts, code_to_stats


def summarize_code(code_id, stats_list, idx_to_label, min_count):
    """One code: count, top labels, mean of 37-dim (ignore FLOW_MISSING)."""
    if len(stats_list) < min_count:
        return None
    counts_per_label = defaultdict(int)
    for _, lab in stats_list:
        counts_per_label[lab] += 1
    top_labels = sorted(counts_per_label.items(), key=lambda x: -x[1])[:5]
    top_labels_named = [(idx_to_label.get(l, str(l)), c) for l, c in top_labels]

    arr = np.array([s for s, _ in stats_list], dtype=np.float64)  # [N, 37]
    mean = np.zeros(37)
    for j in range(37):
        col = arr[:, j]
        valid = col != FLOW_MISSING
        mean[j] = np.mean(col[valid]) if valid.any() else np.nan
    return {
        "code_id": code_id,
        "count": len(stats_list),
        "top_labels": top_labels_named,
        "mean_stats": mean,
    }


def interpret_code(summary, idx_to_label):
    """Heuristic: try to label code as e.g. TCP startup, long flow, short flow."""
    mid = summary["mean_stats"]
    hints = []
    if not np.isnan(mid[27]) and mid[27] >= 0.5:  # syn count
        hints.append("TCP_startup/syn")
    if not np.isnan(mid[0]) and mid[0] > 5000:   # duration_ms
        hints.append("long_flow")
    if not np.isnan(mid[0]) and 0 < mid[0] < 500:
        hints.append("short_flow")
    if not np.isnan(mid[3]) and mid[3] > 1e5:    # total_fwd_bytes
        hints.append("high_fwd_bytes")
    if not np.isnan(mid[4]) and mid[4] > 1e5:
        hints.append("high_bwd_bytes")
    if not np.isnan(mid[9]) and mid[9] > 0 and mid[9] < 1e10 and mid[9] > 1e5:
        hints.append("high_Bps")
    if not np.isnan(mid[18]) and mid[18] > 100:  # mean_iat ms (index 18)
        hints.append("idle_gaps")
    if not np.isnan(mid[34]) and mid[34] > 0:   # mean_active_s (index 34)
        hints.append("active_bursts")
    if summary["top_labels"]:
        dominant = summary["top_labels"][0][0]
        hints.append(f"dom:{dominant[:20]}")
    return "; ".join(hints) if hints else "generic"


def compute_nmi(assignments, labels):
    """NMI between discrete code assignment and label. assignments/labels: 1D int arrays."""
    n = len(assignments)
    if n == 0:
        return 0.0
    codes = np.asarray(assignments, dtype=np.int64)
    labs = np.asarray(labels, dtype=np.int64)
    k_codes = int(codes.max()) + 1
    k_labs = int(labs.max()) + 1
    p_c = np.bincount(codes, minlength=k_codes).astype(np.float64) / n
    p_l = np.bincount(labs, minlength=k_labs).astype(np.float64) / n
    joint = np.zeros((k_codes, k_labs))
    for c, l in zip(codes, labs):
        joint[c, l] += 1.0
    joint /= n
    h_c = -np.sum(p_c * np.log(p_c + 1e-12))
    h_l = -np.sum(p_l * np.log(p_l + 1e-12))
    mi = 0.0
    for c in range(k_codes):
        for l in range(k_labs):
            if joint[c, l] > 0:
                mi += joint[c, l] * np.log(joint[c, l] / (p_c[c] * p_l[l] + 1e-12) + 1e-12)
    if h_c + h_l < 1e-12:
        return 0.0
    return 2.0 * mi / (h_c + h_l)


def _load_from_export(export_dir: Path, K: int):
    """Rebuild (code_counts, code_to_stats) from an export directory."""
    import json as _json
    freq = _json.loads((export_dir / "codebook_freq_data.json").read_text())
    code_counts = np.asarray(freq["code_counts"], dtype=np.int64)
    if len(code_counts) != K:
        raise RuntimeError(f"K mismatch: export={len(code_counts)} vs config={K}")
    arr = np.load(export_dir / "codebook_z_w_labels_codes.npz")
    W = arr["W"]
    L = arr["L"]
    C = arr["C"]
    code_to_stats = defaultdict(list)
    for i in range(len(C)):
        code_to_stats[int(C[i])].append((W[i], int(L[i])))
    return code_counts, code_to_stats


def main():
    args = parse_args()
    set_seed(42)
    config = NetLexiconConfig()
    K = config.vq_codebook_size

    root = Path(config.project_root)
    if args.splits_dir:
        splits_base = root / args.splits_dir
        label_to_idx = load_json(splits_base / config.label_to_idx_path)
        config.num_classes = len(label_to_idx)
    else:
        label_to_idx = load_json(root / config.label_to_idx_path)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    if args.from_export:
        export_dir = Path(args.from_export).resolve()
        print(f"Offline mode: loading code_counts / W / L / C from {export_dir}")
        code_counts, code_to_stats = _load_from_export(export_dir, K)
        manifest_path = export_dir / "codebook_export_manifest.json"
        if manifest_path.is_file():
            m = load_json(manifest_path)
            print(f"  source ckpt: {m.get('ckpt')}")
            print(f"  split: {m.get('split')}  rows_total: {m.get('rows_total')}  epoch: {m.get('checkpoint_epoch')}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint: {args.ckpt}")
        model = NetLexiconPretrainModel(config).to(device)
        ckpt = torch_load_checkpoint(args.ckpt, map_location=device)
        inc = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  epoch {ckpt.get('epoch', '?')}, K={K}")
        if inc.missing_keys:
            print(f"  [strict=False] missing keys: {inc.missing_keys[:6]}{'...' if len(inc.missing_keys)>6 else ''}")
        if inc.unexpected_keys:
            print(f"  [strict=False] unexpected keys:  {inc.unexpected_keys[:6]}{'...' if len(inc.unexpected_keys)>6 else ''}")

        norm_stats = load_json(root / config.norm_stats_path)
        splits = None
        if args.splits_dir:
            splits = load_json(splits_base / config.splits_path)

        train_loader, val_loader, test_loader = build_dataloaders(
            config, norm_stats, label_to_idx, splits=splits
        )
        loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
        n_batches = len(loader) if args.full else args.n_batches
        print(f"Collecting code assignments: {n_batches}/{len(loader)} {args.split} batches ...")
        code_counts, code_to_stats = collect_code_assignments(
            model, loader, n_batches, device, K
        )

    alive = (code_counts >= args.min_count).sum()
    total_assign = code_counts.sum()
    print(f"\n{'='*70}")
    print("[1] Active code overview")
    print(f"{'='*70}")
    print(f"  K (codebook size):      {K}")
    print(f"  alive (count>={args.min_count}): {alive} ({alive/K*100:.1f}%)")
    print(f"  total [WIN] assigns:    {int(total_assign)}")
    probs = code_counts / max(total_assign, 1)
    ppl = np.exp(-np.sum(probs * np.log(probs + 1e-10)))
    print(f"  codebook perplexity:    {ppl:.1f}")

    # Distribution balance diagnostics.
    sorted_counts = np.sort(code_counts)[::-1]
    k_used = int((code_counts > 0).sum())
    if k_used > 0:
        top10_pct = int(np.ceil(K * 0.1))
        top25_pct = int(np.ceil(K * 0.25))
        pct10 = sorted_counts[:top10_pct].sum() / max(total_assign, 1) * 100
        pct25 = sorted_counts[:top25_pct].sum() / max(total_assign, 1) * 100
        alive_counts = code_counts[code_counts >= args.min_count]
        c_min, c_med, c_max = alive_counts.min(), np.median(alive_counts), alive_counts.max()
        print("\n  --- code distribution balance ---")
        print(f"  top 10% codes share:    {pct10:.1f}% (lower is more balanced)")
        print(f"  top 25% codes share:    {pct25:.1f}%")
        print(f"  alive code counts:      min={int(c_min)}  median={int(c_med)}  max={int(c_max)}")

    # [1.5] Code vs label separability.
    all_codes, all_labels = [], []
    purities = []
    for code_id in range(K):
        lst = code_to_stats[code_id]
        if len(lst) < args.min_count:
            continue
        labels_in_code = [lab for _, lab in lst]
        all_codes.extend([code_id] * len(labels_in_code))
        all_labels.extend(labels_in_code)
        cnt = np.bincount(labels_in_code)
        purity = cnt.max() / len(labels_in_code)
        purities.append((purity, len(labels_in_code)))
    if all_codes and all_labels:
        nmi = compute_nmi(np.array(all_codes), np.array(all_labels))
        print(f"\n{'='*70}")
        print("[1.5] Code vs label separability")
        print(f"{'='*70}")
        print(f"  NMI(code, label):       {nmi:.4f} (0~1, higher is better)")
        if purities:
            pur_arr = np.array([p for p, _ in purities])
            w_arr = np.array([w for _, w in purities])
            weighted_purity = np.average(pur_arr, weights=w_arr)
            unweighted_purity = pur_arr.mean()
            print(f"  purity (unweighted):     {unweighted_purity:.4f}")
            print(f"  purity (weighted):       {weighted_purity:.4f}")

    summaries = []
    for code_id in range(K):
        s = summarize_code(code_id, code_to_stats[code_id], idx_to_label, args.min_count)
        if s is not None:
            summaries.append(s)
    summaries.sort(key=lambda x: -x["count"])

    print(f"\n{'='*70}")
    print(f"[2] Top-{args.top_k} active codes: label distribution + window_stats means (subset)")
    print(f"{'='*70}")

    for s in summaries[: args.top_k]:
        interp = interpret_code(s, idx_to_label)
        print(f"\n  --- Code {s['code_id']} (count={s['count']}) ---")
        print(f"  hint: {interp}")
        print(f"  top labels: {s['top_labels']}")
        m = s['mean_stats']
        print("  mean stats: duration_ms={:.0f}  fwd_cnt={:.0f}  bwd_cnt={:.0f}  "
              "total_fwd_B={:.0f}  total_bwd_B={:.0f}  syn={:.1f}  ack={:.1f}  mean_iat_ms={:.0f}".format(
            m[0] if not np.isnan(m[0]) else -1,
            m[1] if not np.isnan(m[1]) else -1,
            m[2] if not np.isnan(m[2]) else -1,
            m[3] if not np.isnan(m[3]) else -1,
            m[4] if not np.isnan(m[4]) else -1,
            m[27] if not np.isnan(m[27]) else -1,   # syn
            m[30] if not np.isnan(m[30]) else -1,   # ack
            m[18] if not np.isnan(m[18]) else -1,   # mean_iat
        ))

    print(f"\n{'='*70}")
    print("[3] Code summaries (heuristic)")
    print(f"{'='*70}")
    for s in summaries[: min(15, len(summaries))]:
        interp = interpret_code(s, idx_to_label)
        print(f"  Code {s['code_id']:3d}  count={s['count']:5d}  ->  {interp}")

    print(f"\n{'='*70}")
    print("[4] window_stats dimension names (37)")
    print(f"{'='*70}")
    for i, name in enumerate(WS_NAMES):
        print(f"  [{i:2d}] {name}")
    print()


if __name__ == "__main__":
    main()
