"""v2 PacketEmbedding: PACKET(14→d) + [SEP] + [WIN] + PE → d_model hidden."""

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TT_PACKET, TT_SEP, TT_WIN


class PacketEmbedding(nn.Module):
    """
    Inputs:
      features:   [B, L, 14] (non-zero only on PACKET positions)
      token_type: [B, L]     int (0=PACKET, 1=SEP, 2=WIN, 3=PAD)

    Returns:
      [B, L, d_model]
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.packet_proj = nn.Linear(config.embed_input_dim, config.d_model)
        self.sep_embed = nn.Parameter(torch.randn(config.d_model) * 0.02)
        self.win_embed = nn.Parameter(torch.randn(config.d_model) * 0.02)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "_positions",
            torch.arange(config.max_seq_len),
            persistent=False,
        )

    def forward(self, features, token_type):
        B, L, _ = features.shape

        h = self.packet_proj(features)  # [B, L, d]

        sep_mask = (token_type == TT_SEP)
        win_mask = (token_type == TT_WIN)

        if sep_mask.any() or win_mask.any():
            h = h.clone()
            if sep_mask.any():
                h[sep_mask] = self.sep_embed.to(h.dtype)
            if win_mask.any():
                h[win_mask] = self.win_embed.to(h.dtype)

        positions = self._positions[:L].unsqueeze(0).expand(B, -1)
        h = h + self.pos_embed(positions)
        h = self.layer_norm(h)
        h = self.dropout(h)
        return h
