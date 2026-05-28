from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbedding(nn.Module):
    def __init__(
        self,
        feature_to_id: Dict[str, int],
        embedding_dim: int = 32,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.feature_to_id = dict(feature_to_id)
        num_features = max(len(self.feature_to_id) + 10, 64)
        self.embedding = nn.Embedding(num_features, embedding_dim)
        self.projection = nn.Linear(embedding_dim + 1, d_model)

    def forward(
        self,
        feature_ids: torch.Tensor,
        feature_values: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.embedding(feature_ids)
        x = torch.cat(
            [emb, feature_values.unsqueeze(-1)], dim=-1
        )
        return self.projection(x)


class DenseGAT(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        d_attn: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_attn = d_attn
        self.W_proj = nn.Linear(d_model, d_attn, bias=False)
        self.a_attn = nn.Parameter(torch.zeros(2 * d_attn, 1))
        nn.init.xavier_uniform_(self.a_attn)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = h.shape
        Wh = self.W_proj(h)

        Wh_i = Wh.unsqueeze(2).expand(B, N, N, self.d_attn)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, self.d_attn)

        Wh_cat = torch.cat([Wh_i, Wh_j], dim=-1)

        e = self.leaky_relu(
            torch.matmul(Wh_cat, self.a_attn).squeeze(-1)
        )

        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)

        z = F.elu(torch.bmm(alpha, Wh))

        return z, alpha


class ODEFunction(nn.Module):
    def __init__(
        self,
        state_dim: int = 1,
        context_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y, z], dim=-1)
        return self.net(x)


class NeuralODE(nn.Module):
    def __init__(self, func: ODEFunction) -> None:
        super().__init__()
        self.func = func

    def forward(
        self,
        y0: torch.Tensor,
        z: torch.Tensor,
        t_span: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        t0, t1 = t_span[0], t_span[1]
        dt = (t1 - t0) / steps
        y = y0

        for _ in range(steps):
            k1 = self.func(y, z)
            k2 = self.func(y + 0.5 * dt * k1, z)
            k3 = self.func(y + 0.5 * dt * k2, z)
            k4 = self.func(y + dt * k3, z)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return y
