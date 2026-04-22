#!/usr/bin/env python3
"""Shared utilities for codebook analysis/plotting/export.

This module centralizes constants and helper functions related to window_stats indexing,
VQ codebook statistics, and (optional) model forward collection.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np


# ---- window_stats metadata ----
# 37-dim window_stats (aligned with prepare/build_dataset.py feature extraction).
WS_NAMES = [
    "duration_ms", "fwd_count", "bwd_count", "total_fwd_bytes", "total_bwd_bytes",
    "mean_fwd_sz", "mean_bwd_sz", "std_fwd_sz", "std_bwd_sz", "bytes_per_sec",
    "mean_sz", "std_sz", "var_sz", "max_sz", "min_sz", "max_fwd_sz", "max_bwd_sz", "skew_sz",
    "mean_iat", "std_iat", "max_iat", "min_iat", "mean_fwd_iat", "std_fwd_iat", "mean_bwd_iat", "std_bwd_iat",
    "fin", "syn", "rst", "psh", "ack", "urg",
    "fwd_pps", "bwd_pps", "mean_active_s", "std_active_s", "mean_idle_s",
]
FLOW_MISSING = -1.0

# Common column indices (by WS_NAMES index).
I_FWD, I_BWD = 1, 2
I_BPS = 9
I_MEAN_SZ = 10
I_MEAN_IAT = 18
I_FIN, I_SYN, I_PSH, I_ACK = 26, 27, 29, 30


# ---- pure numpy helpers ----

def direction_asym(ws_row: np.ndarray) -> float:
    """Directional asymmetry mapped to [0, 1]. 0.5 means balanced."""
    f, b = float(ws_row[I_FWD]), float(ws_row[I_BWD])
    if f + b < 1e-6:
        return 0.5
    return float(np.clip((f - b) / (f + b + 1e-6) * 0.5 + 0.5, 0.0, 1.0))


def dominant_flag(ws_row: np.ndarray) -> int:
    """Dominant TCP flag index: 0=SYN, 1=ACK, 2=PSH, 3=FIN (by max count)."""
    fl = np.array(
        [ws_row[I_SYN], ws_row[I_ACK], ws_row[I_PSH], ws_row[I_FIN]], dtype=np.float64
    )
    if not np.isfinite(fl).all():
        return 0
    return int(np.argmax(fl))


# Backward-compatible aliases (leading underscore).
_direction_asym = direction_asym
_dominant_flag = dominant_flag


def rebuild_code_to_stats(W: np.ndarray, L: np.ndarray, codes_flat: np.ndarray) -> dict:
    """Rebuild {code_id: [(ws_row, label), ...]} from (W, L, codes_flat)."""
    code_to: dict = defaultdict(list)
    for i in range(len(codes_flat)):
        code_to[int(codes_flat[i])].append((W[i], int(L[i])))
    return code_to


# ---- torch/model helpers ----
# These functions import torch lazily so plotting-only usage can work without torch installed.

def load_backbone_weights(model, ckpt_path: Path, device) -> dict:
    """Load a checkpoint into NetLexiconPretrainModel (strict=False to skip extra heads)."""
    import torch  # noqa: F401
    from utils import torch_load_checkpoint

    ckpt = torch_load_checkpoint(str(ckpt_path), map_location=device)
    sd = ckpt["model_state_dict"]
    inc = model.load_state_dict(sd, strict=False)
    return {
        "checkpoint_epoch": ckpt.get("epoch"),
        "missing_keys": list(inc.missing_keys),
        "unexpected_keys": list(inc.unexpected_keys),
    }


def collect_z_codes_stats(
    model,
    val_loader,
    n_batches: Optional[int],
    device,
    K: int,
):
    """Run forward passes and collect (z_e, code_id, win_stats, label) for [WIN] tokens.

    n_batches=None means iterating the full loader.
    Returns: code_counts[K], Z[N,D], W[N,37], L[N], C[N] (C aligned with Z rows).
    """
    import torch
    from tqdm import tqdm
    from config import TT_WIN

    model.eval()
    code_counts = np.zeros(K, dtype=np.int64)
    zs, wss, labs, cds = [], [], [], []

    if n_batches is None:
        n_actual = len(val_loader)
        batch_iter = val_loader
    else:
        n_actual = min(n_batches, len(val_loader))
        batch_iter = islice(val_loader, n_actual)

    with torch.no_grad():
        for batch in tqdm(
            batch_iter,
            total=n_actual,
            desc="collect z/codes",
            unit="batch",
        ):
            (features, tt, vmask, _tt, _ft, _fm,
             win_stats, win_mask, _wc, labels) = batch
            features = features.to(device)
            tt = tt.to(device)
            vmask = vmask.to(device)
            win_stats = win_stats.to(device)
            win_mask = win_mask.to(device)
            labels = labels.to(device)

            h = model.embedding(features, tt)
            h = model.encoder(h, vmask)
            win_flat = tt == TT_WIN
            z_e = h[win_flat]
            if z_e.shape[0] == 0:
                continue

            codebook = model.vq.codebook.weight.float()
            dist = (
                z_e.pow(2).sum(1, keepdim=True)
                - 2 * z_e @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t()
            )
            indices = dist.argmin(dim=1).cpu().numpy()

            ws_flat = win_stats[win_mask].cpu().numpy()
            wl = []
            for b in range(tt.shape[0]):
                for w in range(win_mask.shape[1]):
                    if win_mask[b, w]:
                        wl.append(labels[b].item())
            wl = np.array(wl, dtype=np.int64)

            for cid in indices:
                code_counts[int(cid)] += 1
            zs.append(z_e.cpu().numpy())
            wss.append(ws_flat)
            labs.append(wl)
            cds.append(indices.astype(np.int64))

    if not zs:
        return (
            code_counts,
            np.zeros((0, 1)),
            np.zeros((0, 37)),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    Z = np.concatenate(zs, axis=0)
    W = np.concatenate(wss, axis=0)
    L = np.concatenate(labs, axis=0)
    C = np.concatenate(cds, axis=0)
    return code_counts, Z, W, L, C
