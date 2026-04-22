"""NetLexicon models: pretraining (dual-head + VQ + contrastive) and fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TT_WIN

from .embedding import PacketEmbedding
from .transformer import CausalTransformerEncoder
from .vq import VectorQuantizer
from .prediction_head import (
    TokenTypePredictionHead,
    FeaturePredictionHead,
    ContrastiveProjectionHead,
)


class NetLexiconPretrainModel(nn.Module):
    """Pretraining model: dual-head NTP + VQ on [WIN] + contrastive learning."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = PacketEmbedding(config)
        self.encoder = CausalTransformerEncoder(config)
        self.vq = VectorQuantizer(config)
        self.type_head = TokenTypePredictionHead(config)
        self.feat_head = FeaturePredictionHead(config)
        self.contrast_head = ContrastiveProjectionHead(config)

    def forward(self, features, token_type, valid_mask):
        """
        features:    [B, L, 14]
        token_type:  [B, L]  (0=PACKET, 1=SEP, 2=WIN, 3=PAD)
        valid_mask:  [B, L]  bool

        Returns:
            type_logits: [B, L, 3]
            feat_pred:   [B, L, 7]
            vq_loss:     scalar
            perplexity:  scalar
            z_q_wins:    [N_win, D] (used for contrastive learning)
        """
        h = self.embedding(features, token_type)
        h = self.encoder(h, valid_mask)

        # VQ on [WIN] positions
        win_mask = (token_type == TT_WIN)  # [B, L]
        if win_mask.any():
            z_e_win = h[win_mask]  # [N_win, D]
            z_q, vq_loss, perplexity = self.vq(z_e_win)
            h = h.clone()
            h[win_mask] = z_q
        else:
            z_q = torch.zeros(0, self.config.d_model, device=h.device)
            vq_loss = torch.tensor(0.0, device=h.device)
            perplexity = torch.tensor(1.0, device=h.device)

        type_logits = self.type_head(h)  # [B, L, 3]
        feat_pred = self.feat_head(h)    # [B, L, 7]

        return type_logits, feat_pred, vq_loss, perplexity, z_q

    def compute_loss(self, type_logits, feat_pred, type_target, feat_target, feat_mask,
                     vq_loss, z_q_wins, win_stats, win_mask):
        """
        type_logits: [B, L, 3]
        feat_pred:   [B, L, 7]
        type_target: [B, L] long (-100 = ignore)
        feat_target: [B, L, 7]
        feat_mask:   [B, L] bool
        vq_loss:     scalar
        z_q_wins:    [N_win, D]
        win_stats:   [B, max_wins, 37]
        win_mask:    [B, max_wins] bool

        Returns: total_loss, type_loss, feat_loss, flow_loss
        """
        cfg = self.config

        # Type prediction loss (CrossEntropy, ignore_index=-100)
        B, L, C = type_logits.shape
        type_loss = F.cross_entropy(
            type_logits.reshape(-1, C), type_target.reshape(-1),
            ignore_index=-100
        )

        # Feature prediction loss (MSE, only where feat_mask is True).
        if feat_mask.any():
            feat_loss = (feat_pred[feat_mask] - feat_target[feat_mask]).pow(2).mean()
        else:
            feat_loss = torch.tensor(0.0, device=feat_pred.device)

        # Contrastive loss.
        flow_loss = self._compute_contrastive_loss(z_q_wins, win_stats, win_mask)

        total_loss = (cfg.type_loss_weight * type_loss
                      + cfg.feat_loss_weight * feat_loss
                      + vq_loss
                      + cfg.flow_loss_weight * flow_loss)

        return total_loss, type_loss, feat_loss, flow_loss

    def _compute_contrastive_loss(self, z_q_wins, win_stats, win_mask):
        """InfoNCE between projected z_q and projected window_stats (force fp32 to avoid AMP overflow)."""
        if z_q_wins.shape[0] == 0:
            return torch.tensor(0.0, device=z_q_wins.device
                                if z_q_wins.numel() > 0
                                else win_stats.device)

        flat_stats = win_stats[win_mask]  # [N_win, 37]

        if flat_stats.shape[0] != z_q_wins.shape[0]:
            return torch.tensor(0.0, device=z_q_wins.device)

        N = z_q_wins.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=z_q_wins.device)

        # Force fp32 to avoid exp overflow in half precision.
        z_q_f32 = z_q_wins.float()
        stats_f32 = flat_stats.float()

        r_vq = self.contrast_head.forward_vq(z_q_f32)      # [N, proj_dim]
        r_ws = self.contrast_head.forward_stats(stats_f32)  # [N, proj_dim]

        r_vq = F.normalize(r_vq, dim=-1)
        r_ws = F.normalize(r_ws, dim=-1)

        sim = r_vq @ r_ws.t() / self.config.flow_temperature

        labels = torch.arange(N, device=sim.device)
        loss = F.cross_entropy(sim, labels)
        return loss


class NetLexiconFinetuneModel(nn.Module):
    """Fine-tuning model: mean-pool all [WIN] representations -> classifier."""

    def __init__(self, config, pretrained: NetLexiconPretrainModel, use_vq: bool = True):
        super().__init__()
        self.config = config
        self.use_vq = use_vq
        self.embedding = pretrained.embedding
        self.encoder = pretrained.encoder
        self.vq = pretrained.vq
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.num_classes),
        )

    def forward(self, features, token_type, valid_mask):
        """
        Returns: logits [B, num_classes], vq_loss (scalar; 0 if not using VQ)
        """
        B, L, _ = features.shape
        h = self.embedding(features, token_type)
        h = self.encoder(h, valid_mask)

        win_mask = (token_type == TT_WIN)  # [B, L]
        if win_mask.any() and self.use_vq:
            z_e_win = h[win_mask]
            z_q, vq_loss, _ = self.vq(z_e_win)
            h = h.clone()
            h[win_mask] = z_q
        else:
            vq_loss = torch.tensor(0.0, device=h.device)

        # Mean-pool [WIN] representations per sample (z_q if VQ enabled; else encoder output).
        pooled = torch.zeros(B, self.config.d_model, device=h.device, dtype=h.dtype)
        for i in range(B):
            win_positions = (token_type[i] == TT_WIN)
            if win_positions.any():
                pooled[i] = h[i, win_positions].mean(dim=0)

        logits = self.classifier(pooled)
        return logits, vq_loss
