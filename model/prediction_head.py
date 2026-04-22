"""Prediction heads: token type, feature, and contrastive projection."""

import torch.nn as nn


class TokenTypePredictionHead(nn.Module):
    """Predict next token type (PACKET=0, SEP=1, WIN=2)."""

    def __init__(self, config):
        super().__init__()
        from config import NUM_TOKEN_TYPES
        self.head = nn.Linear(config.d_model, NUM_TOKEN_TYPES)

    def forward(self, h):
        return self.head(h)  # [*, 3]


class FeaturePredictionHead(nn.Module):
    """Predict 7-dim normalized features for the next PACKET token."""

    def __init__(self, config):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.model_feat_dim),
        )

    def forward(self, h):
        return self.head(h)  # [*, 7]


class ContrastiveProjectionHead(nn.Module):
    """Contrastive projection: z_q (d_model) -> proj_dim, window_stats (37) -> proj_dim."""

    def __init__(self, config):
        super().__init__()
        proj_dim = config.contrastive_proj_dim
        self.vq_proj = nn.Sequential(
            nn.Linear(config.d_model, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.stats_ln = nn.LayerNorm(config.window_stats_dim)
        self.stats_proj = nn.Sequential(
            nn.Linear(config.window_stats_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward_vq(self, z_q):
        return self.vq_proj(z_q)

    def forward_stats(self, win_stats):
        x = win_stats.clone()
        x[x == -1] = 0.0
        x = self.stats_ln(x)
        return self.stats_proj(x)
