"""VectorQuantizer with EMA codebook updates (quantizes [WIN] positions)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Input:  z_e [N, D] (all [WIN] hidden states in the batch)
    Output: z_q [N, D], vq_loss (scalar), perplexity (scalar)
    """

    def __init__(self, config):
        super().__init__()
        self.K = config.vq_codebook_size
        self.D = config.vq_dim
        self.beta = config.vq_commitment_weight
        self.use_ema = config.vq_use_ema
        self.decay = config.vq_ema_decay
        self.dead_threshold = config.vq_dead_threshold
        self.revive_every = config.vq_revive_every_n_steps

        self.codebook = nn.Embedding(self.K, self.D)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.1)

        if self.use_ema:
            self.register_buffer("ema_cluster_size", torch.ones(self.K))
            self.register_buffer("ema_embed_sum", self.codebook.weight.clone())
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def _revive_dead_codes(self, flat):
        if not self.use_ema or flat.shape[0] < 10:
            return
        dead = (self.ema_cluster_size < self.dead_threshold).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return
        n_replace = min(len(dead), flat.shape[0])
        idx = torch.randperm(flat.shape[0], device=flat.device)[:n_replace]
        chosen = flat[idx] + torch.randn_like(flat[idx]) * 0.01
        self.codebook.weight.data[dead[:n_replace]] = chosen
        self.ema_embed_sum[dead[:n_replace]] = chosen
        self.ema_cluster_size[dead[:n_replace]] = 1.0

    def forward(self, z_e):
        """
        z_e: [N, D] where N is the total number of [WIN] tokens in the batch.
        """
        z_e = z_e.float()
        N, D = z_e.shape

        dist = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )

        indices = dist.argmin(dim=1)  # [N]
        z_q = self.codebook(indices)  # [N, D]

        enc = F.one_hot(indices, self.K).float()

        if self.training and self.use_ema:
            with torch.no_grad():
                z_e_detached = z_e.detach()
                enc_detached = enc.detach()
                self.ema_cluster_size.mul_(self.decay).add_(
                    enc_detached.sum(0), alpha=1 - self.decay)
                self.ema_embed_sum.mul_(self.decay).add_(
                    enc_detached.t() @ z_e_detached, alpha=1 - self.decay)
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
                self.codebook.weight.data.copy_(
                    self.ema_embed_sum / smoothed.unsqueeze(1)
                )
            self._step.add_(1)
            if self._step.item() % self.revive_every == 0:
                self._revive_dead_codes(z_e.detach())

        vq_loss = self.beta * F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()

        avg_probs = enc.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, vq_loss, perplexity
