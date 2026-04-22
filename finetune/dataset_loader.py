"""Fine-tuning dataloaders (reuses the pretraining NetLexiconDataset + collate_fn)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pretrain.dataset_loader import NetLexiconDataset, collate_fn

from torch.utils.data import DataLoader
from utils import load_json


def build_finetune_dataloaders(config, norm_stats, label_to_idx, splits=None):
    if splits is None:
        splits = load_json(Path(config.project_root) / config.splits_path)

    # Use the shared JSON cache. Use fewer workers for val/test to limit duplicated caches.
    train_ds = NetLexiconDataset(splits["train"], config, norm_stats, label_to_idx)
    val_ds   = NetLexiconDataset(splits["val"],   config, norm_stats, label_to_idx)
    test_ds  = NetLexiconDataset(splits["test"],  config, norm_stats, label_to_idx)

    # Keep workers across epochs to reuse the JSON cache and avoid cold-start I/O spikes.
    train_loader = DataLoader(
        train_ds, batch_size=config.finetune_batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(config.num_workers > 0),
    )
    # Use fewer workers for val/test to limit duplicated caches.
    val_workers = 1 if config.num_workers > 0 else 0
    val_loader = DataLoader(
        val_ds, batch_size=config.finetune_batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(val_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.finetune_batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(val_workers > 0),
    )
    return train_loader, val_loader, test_loader
