"""CausalTransformerEncoder: causal mask + padding mask."""

import torch
import torch.nn as nn


class CausalTransformerEncoder(nn.Module):
    """
    Pre-LN Transformer encoder with causal attention and padding mask.
    """

    def __init__(self, config):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.register_buffer(
            "_causal_mask",
            nn.Transformer.generate_square_subsequent_mask(config.max_seq_len),
            persistent=False,
        )

    def forward(self, h, valid_mask):
        """
        h:          [B, L, d]
        valid_mask: [B, L]  bool, True for valid (non-PAD) tokens

        Returns: [B, L, d]
        """
        L = h.size(1)
        causal_mask = self._causal_mask[:L, :L].to(dtype=h.dtype)
        padding_mask = (~valid_mask).to(dtype=h.dtype)
        padding_mask = padding_mask.masked_fill(padding_mask == 1, float("-inf"))
        return self.encoder(h, mask=causal_mask, src_key_padding_mask=padding_mask)
