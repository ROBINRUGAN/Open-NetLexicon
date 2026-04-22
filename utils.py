"""Common utilities: seeding, normalization, IO."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def torch_load_checkpoint(path, map_location="cpu"):
    """
    torch.load compatibility helper: PyTorch 2.0+ supports weights_only; older versions
    may raise TypeError if the argument is provided.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_packet_features(raw_7: torch.Tensor, norm_stats: dict):
    """
    Normalize raw 7-dim packet features and produce a missingness mask.

    selected_indices = [0, 1, 2, 3, 6, 7, 8]
      0: pkt_size   -> x / 1500
      1: direction  -> keep 0/1
      2: IAT        -> log1p -> z-score
      3: TCP flags  -> x / 255
      4: delta_ipid -> z-score
      5: rel_seq    -> log1p(max(0,x)) -> z-score
      6: rel_ack    -> log1p(max(0,x)) -> z-score

    Returns (normalized, missing_mask) where missing_mask is 1 for missing and 0 otherwise.
    """
    missing_mask = (raw_7 == -1).float()
    x = raw_7.clone()

    x[..., 0] = torch.where(
        raw_7[..., 0] == -1, torch.zeros_like(x[..., 0]), x[..., 0] / 1500.0
    )

    x[..., 1] = torch.where(
        raw_7[..., 1] == -1, torch.zeros_like(x[..., 1]), x[..., 1]
    )

    log_iat = torch.log1p(torch.clamp(x[..., 2], min=0))
    x[..., 2] = torch.where(
        raw_7[..., 2] == -1,
        torch.zeros_like(x[..., 2]),
        (log_iat - norm_stats["iat_log_mean"]) / (norm_stats["iat_log_std"] + 1e-8),
    )

    x[..., 3] = torch.where(
        raw_7[..., 3] == -1, torch.zeros_like(x[..., 3]), x[..., 3] / 255.0
    )

    x[..., 4] = torch.where(
        raw_7[..., 4] == -1,
        torch.zeros_like(x[..., 4]),
        (x[..., 4] - norm_stats["ipid_mean"]) / (norm_stats["ipid_std"] + 1e-8),
    )

    log_seq = torch.log1p(torch.clamp(x[..., 5], min=0))
    x[..., 5] = torch.where(
        raw_7[..., 5] == -1,
        torch.zeros_like(x[..., 5]),
        (log_seq - norm_stats["seq_log_mean"]) / (norm_stats["seq_log_std"] + 1e-8),
    )

    log_ack = torch.log1p(torch.clamp(x[..., 6], min=0))
    x[..., 6] = torch.where(
        raw_7[..., 6] == -1,
        torch.zeros_like(x[..., 6]),
        (log_ack - norm_stats["ack_log_mean"]) / (norm_stats["ack_log_std"] + 1e-8),
    )

    return x, missing_mask


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine annealing with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
