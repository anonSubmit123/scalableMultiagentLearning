from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from core.taskmodel.base import (
    DenseGAT,
    FeatureEmbedding,
    NeuralODE,
    ODEFunction,
)

class GATICDynamicsModel(nn.Module):
    def __init__(
        self,
        ic_type: int,
        feature_names: List[str],
        feature_to_id: Dict[str, int],
        embedding_dim: int = 32,
        d_model: int = 64,
        d_attn: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.ic_type = ic_type
        self.feature_names = list(feature_names)
        self.feature_to_id = dict(feature_to_id)
        self.num_features = len(self.feature_names)

        self.embedding = FeatureEmbedding(
            feature_to_id=self.feature_to_id,
            embedding_dim=embedding_dim,
            d_model=d_model,
        )
        self.gat = DenseGAT(
            d_model=d_model,
            d_attn=d_attn,
        )
        self.ode_func = ODEFunction(
            state_dim=1,
            context_dim=d_attn,
            hidden_dim=hidden_dim,
        )
        self.ode = NeuralODE(self.ode_func)

    def forward(
        self,
        feature_values: torch.Tensor,
        t_span: torch.Tensor,
        steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = feature_values.shape
        device = feature_values.device

        ids_list = [self.feature_to_id[name] for name in self.feature_names]
        feature_ids = (
            torch.tensor(ids_list, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        h = self.embedding(feature_ids, feature_values)
        z, alpha = self.gat(h)

        Z_pooled = torch.mean(z, dim=1)

        y0 = feature_values[:, 0].unsqueeze(-1)
        y_pred = self.ode(y0, Z_pooled, t_span, steps)

        return y_pred, alpha

    def get_active_set(
        self,
        feature_values: torch.Tensor,
        t_span: torch.Tensor,
        steps: int = 10,
        top_k: int = 3,
        gamma: float = 0.5,
    ) -> Tuple[List[str], Dict[str, float], torch.Tensor]:
        B, N = feature_values.shape
        device = feature_values.device

        feature_values_grad = feature_values.clone().detach().requires_grad_(True)

        y_pred, _ = self.forward(feature_values_grad, t_span, steps)

        grad_outputs = torch.ones_like(y_pred)
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=feature_values_grad,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        abs_g = torch.abs(gradients)
        mean_g = torch.mean(abs_g, dim=0)
        var_g = (
            torch.var(abs_g, dim=0) if B > 1 else torch.zeros(N, device=device)
        )
        global_mean = torch.mean(abs_g) + 1e-8
        hybrid_scores = (mean_g + gamma * var_g) / global_mean

        scores_dict = {
            name: float(hybrid_scores[i].item())
            for i, name in enumerate(self.feature_names)
        }

        sorted_indices = torch.argsort(hybrid_scores, descending=True)
        active_indices = sorted_indices[:top_k].tolist()
        active_features = [self.feature_names[idx] for idx in active_indices]

        std_features = (
            torch.std(feature_values, dim=0)
            if B > 1
            else torch.ones(N, device=device)
        )

        H = torch.zeros(B, top_k, top_k, device=device)
        for i_idx, k in enumerate(active_indices):
            g_k = gradients[:, k]
            h_k = torch.autograd.grad(
                outputs=g_k,
                inputs=feature_values_grad,
                grad_outputs=torch.ones_like(g_k),
                retain_graph=True,
                only_inputs=True,
            )[0]

            for j_idx, l in enumerate(active_indices):
                H[:, i_idx, j_idx] = h_k[:, l]

        H_sym = 0.5 * (H + H.transpose(1, 2))
        H_tilde = torch.mean(H_sym, dim=0)
        for i_idx, k in enumerate(active_indices):
            for j_idx, l in enumerate(active_indices):
                H_tilde[i_idx, j_idx] *= std_features[k] * std_features[l]

        return active_features, scores_dict, H_tilde

    def fit(
        self,
        feature_data: torch.Tensor,
        target_trajectories: torch.Tensor,
        epochs: int = 50,
        lr: float = 0.01,
        lambda_sparse: float = 0.001,
    ) -> List[float]:
        device = next(self.parameters()).device
        X = feature_data.to(device)
        Y = target_trajectories.to(device)
        t_span = torch.tensor([0.0, 1.0], dtype=torch.float, device=device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses = []

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            y_pred, alpha = self(X, t_span)

            loss_pred = loss_fn(y_pred, Y)
            loss_sparse = lambda_sparse * torch.sum(torch.abs(alpha))
            loss = loss_pred + loss_sparse

            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        return losses

    def save_weights(self, path: str) -> None:
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        checkpoint = {
            "ic_type": self.ic_type,
            "feature_names": self.feature_names,
            "feature_to_id": self.feature_to_id,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"GATICDynamicsModel saved successfully to: {path}")

    @classmethod
    def load_pretrained(
        cls,
        path: str,
    ) -> GATICDynamicsModel:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained model file not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            ic_type=checkpoint["ic_type"],
            feature_names=checkpoint["feature_names"],
            feature_to_id=checkpoint["feature_to_id"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def get_active_features(self, component: Any) -> list[str]:
        ic_feature_values = []
        for name in self.feature_names:
            val = component.metadata.get(name, 0.5)
            ic_feature_values.append(float(val))

        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        ic_tensor = torch.tensor([ic_feature_values], dtype=torch.float, device=device)
        t_span = torch.tensor([0.0, 1.0], dtype=torch.float, device=device)

        active_set, _, _ = self.get_active_set(
            ic_tensor, t_span, top_k=min(3, len(self.feature_names))
        )
        return active_set

    def predict_demand(
        self,
        component: Any,
        t_span: torch.Tensor | None = None,
        steps: int = 10,
    ) -> float:
        ic_feature_values = []
        for i, name in enumerate(self.feature_names):
            if i == 0 and hasattr(component, "demand"):
                val = component.demand
            else:
                val = component.metadata.get(name, 0.5)
            ic_feature_values.append(float(val))

        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        ic_tensor = torch.tensor([ic_feature_values], dtype=torch.float, device=device)

        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], dtype=torch.float, device=device)
        else:
            t_span = t_span.to(device)

        with torch.no_grad():
            y_pred, _ = self.forward(ic_tensor, t_span, steps)

        predicted_val = float(y_pred.item())
        return max(0.0, predicted_val)


import typing
if typing.TYPE_CHECKING:
    from runtime.factory import ICDynamicsModel
_istype_GATICDynamicsModel: typing.Type[ICDynamicsModel] = GATICDynamicsModel
