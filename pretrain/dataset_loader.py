"""
Pretraining dataset: variable-length bursts with [SEP] and [WIN] separators.

Returns:
  features:    [L, 14]
  token_type:  [L] (0=PACKET, 1=SEP, 2=WIN, 3=PAD)
  valid_mask:  [L] (True for non-PAD)
  type_target: [L] (token_type[i+1], last valid position is -100)
  feat_target: [L, 7] (only when i+1 is PACKET)
  feat_mask:   [L] (True where feat loss applies)
  win_stats:   [max_wins, 37]
  win_count:   int
  label:       int
"""

import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NetLexiconConfig, TT_PACKET, TT_SEP, TT_WIN, TT_PAD
from utils import normalize_packet_features, load_json


_JSON_CACHE_HARD_CAP = int(os.environ.get("NETLEX_JSON_CACHE", "300"))


class NetLexiconDataset(Dataset):
    def __init__(self, split_entries, config, norm_stats, label_to_idx,
                 cache_size: int | None = None):
        self.config = config
        self.norm_stats = norm_stats
        self.label_to_idx = label_to_idx
        self._json_cache: "OrderedDict[str, dict]" = OrderedDict()

        self.entries = []
        unique_files: set[str] = set()
        for entry in split_entries:
            self.entries.append((
                entry["file"],
                entry["flow_idx"],
                entry["num_windows"],
            ))
            unique_files.add(entry["file"])

        # Fit all unique files if possible; otherwise fall back to hard cap.
        hard_cap = _JSON_CACHE_HARD_CAP if cache_size is None else cache_size
        self._cache_size = min(hard_cap, max(1, len(unique_files)))

    def __len__(self):
        return len(self.entries)

    def _load_json(self, rel_path):
        cache = self._json_cache
        hit = cache.get(rel_path)
        if hit is not None:
            cache.move_to_end(rel_path)
            return hit
        full = Path(self.config.project_root) / self.config.dataset_dir / rel_path
        with open(full, encoding="utf-8") as f:
            obj = json.load(f)
        cache[rel_path] = obj
        if len(cache) > self._cache_size:
            cache.popitem(last=False)
        return obj

    def _extract_raw_features(self, pkt):
        return [float(pkt[si]) for si in self.config.selected_indices]

    def __getitem__(self, idx):
        json_rel, flow_idx, num_windows = self.entries[idx]
        data = self._load_json(json_rel)
        flow = data["flows"][flow_idx]
        label = self.label_to_idx[data["label"]]

        packets = flow["packets"]
        bursts = flow["bursts"]
        flow_win_stats = flow["window_stats"]
        cfg = self.config
        L = cfg.max_seq_len
        BPW = cfg.bursts_per_window

        token_feats = []     # list of (7-dim raw features or None for SEP/WIN)
        token_types = []     # list of int
        win_stats_list = []  # list of 37-dim vectors for each [WIN]
        burst_in_win = 0     # bursts processed in current window
        win_start_burst = 0  # starting burst index of current window

        # window_stats is a stride-1 sliding window:
        #   window_stats[i] corresponds to bursts[i..i+BPW)
        # We insert [WIN] using stride=BPW (non-overlapping):
        #   1st [WIN] after bursts 0..BPW-1 -> window_stats[0]
        #   2nd [WIN] after bursts BPW..2*BPW-1 -> window_stats[BPW]
        #   k-th [WIN] -> window_stats[k * BPW]

        n_bursts = len(bursts)
        for bi in range(n_bursts):
            b_start = bursts[bi]
            b_end = bursts[bi + 1] if (bi + 1) < n_bursts else len(packets)
            burst_pkts = packets[b_start:b_end]

            for pkt in burst_pkts:
                if len(token_feats) >= L:
                    break
                token_feats.append(self._extract_raw_features(pkt))
                token_types.append(TT_PACKET)

            if len(token_feats) >= L:
                break

            # End of burst: add [SEP].
            token_feats.append(None)
            token_types.append(TT_SEP)
            burst_in_win += 1

            if len(token_feats) >= L:
                break

            # Every BPW bursts: add [WIN].
            if burst_in_win == BPW:
                token_feats.append(None)
                token_types.append(TT_WIN)

                ws_idx = win_start_burst
                if ws_idx < len(flow_win_stats):
                    win_stats_list.append(flow_win_stats[ws_idx])
                else:
                    win_stats_list.append([0.0] * cfg.window_stats_dim)

                burst_in_win = 0
                win_start_burst += BPW

                if len(token_feats) >= L:
                    break

        actual_len = min(len(token_feats), L)
        token_feats = token_feats[:actual_len]
        token_types = token_types[:actual_len]

        # Build raw_7 tensor.
        raw_7 = torch.zeros(L, cfg.model_feat_dim)
        for i, (feat, tt) in enumerate(zip(token_feats, token_types)):
            if tt == TT_PACKET and feat is not None:
                raw_7[i] = torch.tensor(feat, dtype=torch.float32)

        # Normalize.
        normalized, missing = normalize_packet_features(raw_7, self.norm_stats)
        for i in range(actual_len):
            if token_types[i] != TT_PACKET:
                normalized[i] = 0.0
                missing[i] = 0.0
        normalized[actual_len:] = 0.0
        missing[actual_len:] = 0.0

        features = torch.cat([normalized, missing], dim=-1)  # [L, 14]

        # token_type tensor
        tt_tensor = torch.full((L,), TT_PAD, dtype=torch.long)
        for i, tt in enumerate(token_types):
            tt_tensor[i] = tt

        # valid_mask
        valid_mask = torch.zeros(L, dtype=torch.bool)
        valid_mask[:actual_len] = True

        # Targets & masks for the dual-head objective.
        # type_target: position i -> token_type[i+1], invalid positions use -100 (ignore_index)
        type_target = torch.full((L,), -100, dtype=torch.long)
        for i in range(actual_len - 1):
            next_tt = token_types[i + 1]
            type_target[i] = next_tt  # token types are already encoded as 0/1/2

        # feat_target & feat_mask: valid only when i+1 is PACKET.
        feat_target = torch.zeros(L, cfg.model_feat_dim)
        feat_mask = torch.zeros(L, dtype=torch.bool)
        for i in range(actual_len - 1):
            if token_types[i + 1] == TT_PACKET:
                feat_target[i] = normalized[i + 1]
                feat_mask[i] = True

        # win_stats tensor: [max_wins, 37]
        max_wins = L // 10  # loose upper bound
        win_count = len(win_stats_list)
        win_stats = torch.zeros(max_wins, cfg.window_stats_dim)
        for i, ws in enumerate(win_stats_list):
            if i < max_wins:
                win_stats[i] = torch.tensor(ws, dtype=torch.float32)

        return (features, tt_tensor, valid_mask,
                type_target, feat_target, feat_mask,
                win_stats, win_count, label)


def collate_fn(batch):
    """Custom collate to handle variable-length win_stats."""
    (features_list, tt_list, vmask_list,
     type_target_list, feat_target_list, feat_mask_list,
     win_stats_list, win_count_list, label_list) = zip(*batch)

    features = torch.stack(features_list)
    tt = torch.stack(tt_list)
    vmask = torch.stack(vmask_list)
    type_target = torch.stack(type_target_list)
    feat_target = torch.stack(feat_target_list)
    feat_mask = torch.stack(feat_mask_list)
    labels = torch.tensor(label_list, dtype=torch.long)

    # Align win_stats to the max win_count in this batch.
    max_wc = max(win_count_list) if max(win_count_list) > 0 else 1
    B = len(batch)
    ws_dim = features_list[0].shape[-1] // 2  # 14 // 2 = 7? NO
    ws_dim = win_stats_list[0].shape[-1]  # 37
    win_stats = torch.zeros(B, max_wc, ws_dim)
    win_mask = torch.zeros(B, max_wc, dtype=torch.bool)
    for i, (ws, wc) in enumerate(zip(win_stats_list, win_count_list)):
        if wc > 0:
            win_stats[i, :wc] = ws[:wc]
            win_mask[i, :wc] = True

    win_counts = torch.tensor(win_count_list, dtype=torch.long)

    return (features, tt, vmask,
            type_target, feat_target, feat_mask,
            win_stats, win_mask, win_counts, labels)


def build_dataloaders(config, norm_stats, label_to_idx, splits=None):
    """Build train/val/test DataLoaders."""
    if splits is None:
        splits = load_json(Path(config.project_root) / config.splits_path)

    # Use the adaptive cache size (<= _JSON_CACHE_HARD_CAP) and keep it across epochs.
    train_ds = NetLexiconDataset(splits["train"], config, norm_stats, label_to_idx)
    val_ds   = NetLexiconDataset(splits["val"],   config, norm_stats, label_to_idx)
    test_ds  = NetLexiconDataset(splits["test"],  config, norm_stats, label_to_idx)

    # Keep workers across epochs to reuse the JSON cache and avoid cold-start I/O spikes.
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(config.num_workers > 0),
    )
    # Use fewer workers for val/test to limit duplicated caches.
    val_workers = 1 if config.num_workers > 0 else 0
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(val_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(val_workers > 0),
    )
    return train_loader, val_loader, test_loader
