"""NetLexicon global configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ===== Token type constants =====
TT_PACKET = 0
TT_SEP = 1
TT_WIN = 2
TT_PAD = 3
NUM_TOKEN_TYPES = 3  # output classes (PACKET/SEP/WIN), excluding PAD


@dataclass
class NetLexiconConfig:
    # ===== Paths =====
    project_root: str = ""
    dataset_dir: str = "dataset"
    norm_stats_path: str = "norm_stats.json"
    splits_path: str = "splits.json"
    label_to_idx_path: str = "label_to_idx.json"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ===== Data =====
    raw_feat_dim: int = 11
    selected_indices: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 6, 7, 8]
    )
    model_feat_dim: int = 7
    embed_input_dim: int = 14          # 7 features + 7 missingness indicators
    max_seq_len: int = 256
    num_classes: int = 42
    bursts_per_window: int = 5         # bursts per window
    window_stats_dim: int = 37         # window_stats dimension

    # ===== Model =====
    # Defaults match the lightweight config used by training scripts (~2M params).
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 16
    d_ff: int = 768
    dropout: float = 0.1

    # ===== VQ =====
    vq_codebook_size: int = 128
    vq_dim: int = 192
    vq_ema_decay: float = 0.99
    vq_commitment_weight: float = 0.25
    vq_use_ema: bool = True
    vq_dead_threshold: float = 0.01
    vq_revive_every_n_steps: int = 500

    # ===== Pretraining =====
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 20
    gradient_clip: float = 1.0
    num_workers: int = 4
    use_amp: bool = True
    log_every_n_steps: int = 100
    val_every_n_epochs: int = 1
    save_top_k: int = 3

    # ===== Pretraining loss weights =====
    type_loss_weight: float = 1.0      # token type CE weight (set 0 for ablation)
    feat_loss_weight: float = 1.0      # feature MSE weight
    flow_loss_weight: float = 0.1      # contrastive loss weight (set 0 for ablation)
    flow_temperature: float = 0.07     # InfoNCE temperature
    contrastive_proj_dim: int = 64     # contrastive projection dim

    # ===== Fine-tuning =====
    finetune_lr: float = 5e-5
    finetune_epochs: int = 20
    finetune_batch_size: int = 128
    freeze_encoder_epochs: int = 5

    # ===== Splits =====
    split_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def __post_init__(self):
        if not self.project_root:
            self.project_root = str(Path(__file__).resolve().parent)
