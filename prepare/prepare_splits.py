#!/usr/bin/env python3
"""
Prepare stratified train/val/test splits on flow level.

Outputs:
  - splits.json
  - label_to_idx.json

Usage:
  python prepare/prepare_splits.py
  python prepare/prepare_splits.py --dataset CipherSpectrum
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42
BURSTS_PER_WIN = 5
MIN_WINDOWS = 1


def apply_per_class_flow_balance(
    label_flows: dict[str, list],
    min_flows: int | None,
    max_flows: int | None,
    seed: int,
) -> dict[str, list]:
    """
    Per-class flow balancing (applied before splitting):
    - drop classes with fewer than min_flows
    - randomly cap classes to max_flows
    """
    if min_flows is None and max_flows is None:
        return label_flows

    rng = random.Random(seed)
    new_flows: dict[str, list] = {}
    dropped: list[tuple[str, int]] = []
    capped: list[tuple[str, int, int]] = []

    for label in sorted(label_flows.keys()):
        flows = list(label_flows[label])
        n = len(flows)
        if min_flows is not None and n < min_flows:
            dropped.append((label, n))
            continue
        if max_flows is not None and n > max_flows:
            rng.shuffle(flows)
            flows = flows[:max_flows]
            capped.append((label, n, len(flows)))
        new_flows[label] = flows

    print("\n[per-class balance]")
    if min_flows is not None:
        print(f"  min_flows_per_class={min_flows} | dropped_classes={len(dropped)}")
        for lbl, n in dropped[:20]:
            print(f"    drop: {lbl} (flows={n})")
        if len(dropped) > 20:
            print(f"    ... plus {len(dropped) - 20} more")
    if max_flows is not None:
        print(f"  max_flows_per_class={max_flows} | capped_classes={len(capped)}")
        for lbl, n_before, n_after in capped[:15]:
            print(f"    cap: {lbl}  {n_before} -> {n_after}")
        if len(capped) > 15:
            print(f"    ... plus {len(capped) - 15} more")
    print(f"  num_classes_after={len(new_flows)}")

    return new_flows


def apply_dataset_flow_cap(
    label_flows: dict[str, list],
    max_flows: int,
    dataset_name: str,
) -> dict[str, list]:
    """
    Cap total flows per dataset by preferring longer flows (higher num_windows).
    """
    all_flows = []
    for label, flows in label_flows.items():
        for f in flows:
            all_flows.append((label, f))

    n_before = len(all_flows)
    if n_before <= max_flows:
        return label_flows

    # Sort by num_windows (desc). Stable sort keeps original order for ties.
    all_flows.sort(key=lambda x: x[1]["num_windows"], reverse=True)
    selected = all_flows[:max_flows]

    new_flows: dict[str, list] = {}
    for label, f in selected:
        new_flows.setdefault(label, []).append(f)

    n_after = len(selected)
    print(
        f"\n[per-dataset cap] {dataset_name}: {n_before} -> {n_after} flows "
        f"(keep top {max_flows} by num_windows)"
    )
    print(f"  classes_kept: {len(new_flows)} / {len(label_flows)}")

    return new_flows


def parse_args():
    parser = argparse.ArgumentParser(description="Flow-level stratified split.")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="One or more dataset paths relative to dataset/. "
                             "Example: --dataset CipherSpectrum  OR  --dataset ISCXVPN2016/NonVPN MAWI VisQUIC. "
                             "Default: scan all.")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Output directory name under splits/. Required when multiple datasets are provided. "
                             "For a single dataset, defaults to the dataset path.")
    parser.add_argument("--min_flows_per_class", type=int, default=None,
                        help="Minimum flows per class; classes below this threshold are dropped.")
    parser.add_argument("--max_flows_per_class", type=int, default=None,
                        help="Maximum flows per class; classes above this threshold are randomly capped.")
    parser.add_argument("--max_flows_per_dataset", type=int, default=None,
                        help="(Multi-dataset) maximum total flows per dataset; prefers longer flows by num_windows.")
    return parser.parse_args()


def do_split(
    scan_dir: Path,
    out_dir: Path,
    dataset_name: str | None,
    min_flows_per_class: int | None = None,
    max_flows_per_class: int | None = None,
):
    """Scan -> split -> save for a single dataset."""
    if not scan_dir.exists():
        print(f"Dataset dir does not exist: {scan_dir}")
        sys.exit(1)

    label_flows = defaultdict(list)
    total_flows = 0
    skipped_flows = 0

    json_files = sorted(scan_dir.rglob("*.json"))
    scope = dataset_name if dataset_name else "all"
    print(f"[{scope}] JSON files: {len(json_files)} (dir: {scan_dir})")

    for jf in json_files:
        rel_path = str(jf.relative_to(DATASET_DIR))
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        label = data["label"]

        for flow_idx, flow in enumerate(data["flows"]):
            total_flows += 1
            n_bursts = len(flow["bursts"])
            n_windows = n_bursts // BURSTS_PER_WIN
            if n_windows < MIN_WINDOWS:
                skipped_flows += 1
                continue
            label_flows[label].append({
                "file": rel_path,
                "flow_idx": flow_idx,
                "num_windows": n_windows,
            })

    print(
        f"flows_total={total_flows}, flows_valid={total_flows - skipped_flows}, "
        f"skipped(<{MIN_WINDOWS} win)={skipped_flows}"
    )
    print(f"num_classes={len(label_flows)}")

    if not label_flows:
        print("No valid flows. Skip.")
        return

    label_flows = apply_per_class_flow_balance(
        dict(label_flows),
        min_flows_per_class,
        max_flows_per_class,
        SEED,
    )

    if not label_flows:
        print("No classes left after balancing. Skip.")
        return

    labels_sorted = sorted(label_flows.keys())
    label_to_idx = {name: i for i, name in enumerate(labels_sorted)}

    random.seed(SEED)
    train_entries, val_entries, test_entries = [], [], []

    for label in labels_sorted:
        flows = label_flows[label]
        random.shuffle(flows)
        n = len(flows)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        train_entries.extend(flows[:n_train])
        val_entries.extend(flows[n_train:n_train + n_val])
        test_entries.extend(flows[n_train + n_val:])

    splits = {"train": train_entries, "val": val_entries, "test": test_entries}

    total_wins = sum(e["num_windows"] for e in train_entries + val_entries + test_entries)
    print("\nSplit summary:")
    print(f"  train: {len(train_entries)} flows")
    print(f"  val:   {len(val_entries)} flows")
    print(f"  test:  {len(test_entries)} flows")
    print(f"  windows_total: {total_wins}")

    out_dir.mkdir(parents=True, exist_ok=True)

    splits_path = out_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False)
    print(f"\nSaved: {splits_path}")

    lti_path = out_dir / "label_to_idx.json"
    with open(lti_path, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Saved: {lti_path}")


def do_split_multi(
    scan_dirs: list[Path],
    out_dir: Path,
    out_name: str,
    dataset_names: list[str],
    min_flows_per_class: int | None = None,
    max_flows_per_class: int | None = None,
    max_flows_per_dataset: int | None = None,
):
    """Scan/split/save for multiple datasets and merge by label."""
    # Keep per-dataset flows to support optional per-dataset flow caps.
    per_dataset_flows: list[dict[str, list]] = []
    total_flows = 0
    skipped_flows = 0
    total_json = 0

    for scan_dir, ds_name in zip(scan_dirs, dataset_names):
        json_files = sorted(scan_dir.rglob("*.json"))
        total_json += len(json_files)
        print(f"  [{ds_name}] JSON files: {len(json_files)} (dir: {scan_dir})")

        ds_label_flows: dict[str, list] = defaultdict(list)
        for jf in json_files:
            rel_path = str(jf.relative_to(DATASET_DIR))
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            label = data["label"]

            for flow_idx, flow in enumerate(data["flows"]):
                total_flows += 1
                n_bursts = len(flow["bursts"])
                n_windows = n_bursts // BURSTS_PER_WIN
                if n_windows < MIN_WINDOWS:
                    skipped_flows += 1
                    continue
                ds_label_flows[label].append({
                    "file": rel_path,
                    "flow_idx": flow_idx,
                    "num_windows": n_windows,
                })

        ds_flows = dict(ds_label_flows)
        ds_total = sum(len(v) for v in ds_flows.values())
        print(f"    flows_valid={ds_total}, num_classes={len(ds_flows)}")

        # Optional per-dataset cap (prefer longer flows).
        if max_flows_per_dataset is not None:
            ds_flows = apply_dataset_flow_cap(ds_flows, max_flows_per_dataset, ds_name)

        per_dataset_flows.append(ds_flows)

    # Merge all datasets.
    label_flows: dict[str, list] = defaultdict(list)
    for ds_flows in per_dataset_flows:
        for label, flows in ds_flows.items():
            label_flows[label].extend(flows)

    print(f"\n[{out_name}] JSON files: {total_json}")
    print(
        f"flows_total={total_flows}, flows_valid={total_flows - skipped_flows}, "
        f"skipped(<{MIN_WINDOWS} win)={skipped_flows}"
    )
    print(
        f"merged_num_classes={len(label_flows)}, merged_flows={sum(len(v) for v in label_flows.values())}"
    )

    if not label_flows:
        print("No valid flows. Skip.")
        return

    label_flows = apply_per_class_flow_balance(
        dict(label_flows),
        min_flows_per_class,
        max_flows_per_class,
        SEED,
    )

    if not label_flows:
        print("No classes left after balancing. Skip.")
        return

    labels_sorted = sorted(label_flows.keys())
    label_to_idx = {name: i for i, name in enumerate(labels_sorted)}

    random.seed(SEED)
    train_entries, val_entries, test_entries = [], [], []

    for label in labels_sorted:
        flows = label_flows[label]
        random.shuffle(flows)
        n = len(flows)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        train_entries.extend(flows[:n_train])
        val_entries.extend(flows[n_train:n_train + n_val])
        test_entries.extend(flows[n_train + n_val:])

    splits = {"train": train_entries, "val": val_entries, "test": test_entries}

    total_wins = sum(e["num_windows"] for e in train_entries + val_entries + test_entries)
    print("\nSplit summary:")
    print(f"  train: {len(train_entries)} flows")
    print(f"  val:   {len(val_entries)} flows")
    print(f"  test:  {len(test_entries)} flows")
    print(f"  windows_total: {total_wins}")

    out_dir.mkdir(parents=True, exist_ok=True)

    splits_path = out_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False)
    print(f"\nSaved: {splits_path}")

    lti_path = out_dir / "label_to_idx.json"
    with open(lti_path, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Saved: {lti_path}")


def main():
    args = parse_args()

    if (
        args.min_flows_per_class is not None
        and args.max_flows_per_class is not None
        and args.min_flows_per_class > args.max_flows_per_class
    ):
        print("Error: --min_flows_per_class cannot be greater than --max_flows_per_class", file=sys.stderr)
        sys.exit(1)

    mf, xf, xd = args.min_flows_per_class, args.max_flows_per_class, args.max_flows_per_dataset

    if args.dataset:
        if len(args.dataset) == 1:
            ds = args.dataset[0]
            scan_dir = DATASET_DIR / ds
            out_name = args.out_name or ds.replace("/", "_")
            out_dir = PROJECT_ROOT / "splits" / out_name
            do_split(scan_dir, out_dir, ds, mf, xf)
        else:
            if not args.out_name:
                print("Error: --out_name is required when using multiple datasets", file=sys.stderr)
                sys.exit(1)
            scan_dirs = [DATASET_DIR / ds for ds in args.dataset]
            for sd in scan_dirs:
                if not sd.exists():
                    print(f"Dataset dir does not exist: {sd}", file=sys.stderr)
                    sys.exit(1)
            out_dir = PROJECT_ROOT / "splits" / args.out_name
            do_split_multi(scan_dirs, out_dir, args.out_name, args.dataset, mf, xf, xd)
    else:
        scan_dir = DATASET_DIR
        out_dir = PROJECT_ROOT
        do_split(scan_dir, out_dir, None, mf, xf)


if __name__ == "__main__":
    main()
