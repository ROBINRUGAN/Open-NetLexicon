#!/usr/bin/env python3
"""
Compute normalization statistics (norm_stats.json) by scanning all JSON files under dataset/.
Uses Welford's online algorithm and does not require splits.json.
"""

import json
import math
import sys
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


class WelfordAccumulator:
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return 0.0
        return math.sqrt(self.M2 / self.n)


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset dir does not exist: {DATASET_DIR}")
        sys.exit(1)

    json_files = sorted(DATASET_DIR.rglob("*.json"))
    print(f"JSON files: {len(json_files)} (dir: {DATASET_DIR})")

    acc_iat = WelfordAccumulator()
    acc_ipid = WelfordAccumulator()
    acc_seq = WelfordAccumulator()
    acc_ack = WelfordAccumulator()

    total_packets = 0
    total_flows = 0

    for jf in tqdm(json_files, desc="JSON", unit="file"):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        for flow in data["flows"]:
            total_flows += 1
            for pkt in flow["packets"]:
                total_packets += 1

                iat = pkt[2]
                if iat != -1:
                    acc_iat.update(math.log1p(max(0.0, iat)))

                ipid = pkt[6]
                if ipid != -1:
                    acc_ipid.update(float(ipid))

                seq = pkt[7]
                if seq != -1:
                    acc_seq.update(math.log1p(max(0.0, float(seq))))

                ack = pkt[8]
                if ack != -1:
                    acc_ack.update(math.log1p(max(0.0, float(ack))))

    norm_stats = {
        "iat_log_mean": acc_iat.mean,
        "iat_log_std": acc_iat.std,
        "ipid_mean": acc_ipid.mean,
        "ipid_std": acc_ipid.std,
        "seq_log_mean": acc_seq.mean,
        "seq_log_std": acc_seq.std,
        "ack_log_mean": acc_ack.mean,
        "ack_log_std": acc_ack.std,
    }

    print(f"\nJSON total: {len(json_files)}")
    print(f"Flows total: {total_flows}")
    print(f"Packets total: {total_packets}")
    for k, v in norm_stats.items():
        print(f"  {k}: {v:.6f}")

    out_path = PROJECT_ROOT / "norm_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
