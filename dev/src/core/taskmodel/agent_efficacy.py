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


class GATAgentEfficacyModel(nn.Module):
    def __init__(
        self,
        ic_type: int,
        active_feature_names: List[str],
        agent_feature_names: List[str],
        feature_to_id: Dict[str, int],
        agent_capability: str | None = None,
        embedding_dim: int = 32,
        d_model: int = 64,
        d_attn: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.ic_type = ic_type
        self.active_feature_names = list(active_feature_names)
        self.agent_feature_names = list(agent_feature_names)
        self.feature_to_id = dict(feature_to_id)
        self.agent_capability = agent_capability
        self.all_features = self.active_feature_names + self.agent_feature_names

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
        active_ic_feature_values: torch.Tensor,
        agent_feature_values: torch.Tensor,
        t_span: torch.Tensor,
        steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = active_ic_feature_values.shape[0]
        device = active_ic_feature_values.device

        features_concat = torch.cat(
            [active_ic_feature_values, agent_feature_values], dim=-1
        )

        ids_list = [self.feature_to_id[name] for name in self.all_features]
        feature_ids = (
            torch.tensor(ids_list, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        h = self.embedding(feature_ids, features_concat)
        z, alpha = self.gat(h)

        Z_pooled = torch.mean(z, dim=1)

        y0 = torch.ones(B, 1, device=device)
        t0, t1 = t_span[0], t_span[1]
        dt = (t1 - t0) / steps

        trajectory = [y0]
        y = y0
        for _ in range(steps):
            k1 = self.ode.func(y, Z_pooled)
            k2 = self.ode.func(y + 0.5 * dt * k1, Z_pooled)
            k3 = self.ode.func(y + 0.5 * dt * k2, Z_pooled)
            k4 = self.ode.func(y + dt * k3, Z_pooled)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            trajectory.append(y)

        traj_tensor = torch.cat(trajectory, dim=-1)

        return traj_tensor, alpha

    def _do_predict_efficacy(
        self,
        active_ic_feature_values: torch.Tensor,
        agent_feature_values: torch.Tensor,
        t_span: torch.Tensor,
        steps: int = 10,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = active_ic_feature_values.shape[0]
        device = active_ic_feature_values.device

        active_ic_grad = active_ic_feature_values.clone().detach().requires_grad_(True)
        agent_grad = agent_feature_values.clone().detach().requires_grad_(True)

        traj, alpha = self.forward(active_ic_grad, agent_grad, t_span, steps)

        eta_min, _ = torch.min(traj, dim=-1)
        eta_min = eta_min.unsqueeze(-1)

        grad_outputs = torch.ones_like(eta_min)
        g_ic = torch.autograd.grad(
            outputs=eta_min,
            inputs=active_ic_grad,
            grad_outputs=grad_outputs,
            retain_graph=True,
            allow_unused=True,
        )[0]
        g_agent = torch.autograd.grad(
            outputs=eta_min,
            inputs=agent_grad,
            grad_outputs=grad_outputs,
            retain_graph=True,
            allow_unused=True,
        )[0]

        Phi: Dict[str, float] = {}

        if g_ic is not None:
            mean_ic = torch.mean(torch.abs(g_ic), dim=0)
            for i, name in enumerate(self.active_feature_names):
                Phi[f"gradient_{name}"] = float(mean_ic[i].item())

        if g_agent is not None:
            mean_agent = torch.mean(torch.abs(g_agent), dim=0)
            for i, name in enumerate(self.agent_feature_names):
                Phi[f"gradient_{name}"] = float(mean_agent[i].item())

        mean_alpha = torch.mean(alpha, dim=0)
        for i, name_i in enumerate(self.all_features):
            for j, name_j in enumerate(self.all_features):
                if i != j and mean_alpha[i, j] > 0.05:
                    Phi[f"attention_{name_j}_to_{name_i}"] = float(
                        mean_alpha[i, j].item()
                    )

        return eta_min.detach(), Phi

    def compute_suppression_ratio(
        self,
        active_ic_feature_values: torch.Tensor,
        agent_feature_values: torch.Tensor,
        reference_ic_feature_values: torch.Tensor,
        t_span: torch.Tensor,
        steps: int = 10,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        eta_current, _ = self._do_predict_efficacy(
            active_ic_feature_values, agent_feature_values, t_span, steps
        )
        eta_ref, _ = self._do_predict_efficacy(
            reference_ic_feature_values, agent_feature_values, t_span, steps
        )

        ratio = 1.0 - eta_current / (eta_ref + epsilon)
        return torch.clamp(ratio, 0.0, 1.0)

    def fit(
        self,
        active_ic_data: torch.Tensor,
        agent_data: torch.Tensor,
        target_efficacy: torch.Tensor,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> List[float]:
        device = next(self.parameters()).device
        X_ic = active_ic_data.to(device)
        X_agent = agent_data.to(device)
        Y = target_efficacy.to(device)
        t_span = torch.tensor([0.0, 1.0], dtype=torch.float, device=device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses = []

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            traj, _ = self(X_ic, X_agent, t_span)

            eta_min, _ = torch.min(traj, dim=-1)
            eta_min = eta_min.unsqueeze(-1)

            loss = loss_fn(eta_min, Y)
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
            "active_feature_names": self.active_feature_names,
            "agent_feature_names": self.agent_feature_names,
            "feature_to_id": self.feature_to_id,
            "agent_capability": self.agent_capability,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"GATAgentEfficacyModel saved successfully to: {path}")

    @classmethod
    def load_pretrained(
        cls,
        path: str,
    ) -> GATAgentEfficacyModel:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained model file not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            ic_type=checkpoint["ic_type"],
            active_feature_names=checkpoint["active_feature_names"],
            agent_feature_names=checkpoint["agent_feature_names"],
            feature_to_id=checkpoint["feature_to_id"],
            agent_capability=checkpoint.get("agent_capability", None),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def predict_efficacy(
        self,
        agent: Any,
        component: Any,
        config: Any = None,
    ) -> float:
        active_set = self.active_feature_names

        active_values = []
        for name in active_set:
            val = component.metadata.get(name, 0.5)
            active_values.append(float(val))

        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        active_tensor = torch.tensor([active_values], dtype=torch.float, device=device)
        t_span = torch.tensor([0.0, 1.0], dtype=torch.float, device=device)

        agent_feature_names = self.agent_feature_names
        agent_values = []

        agent_spec = None
        if config and hasattr(config, "agents") and agent.agent_id in config.agents:
            agent_spec = config.agents[agent.agent_id]

        for name in agent_feature_names:
            val = 0.5
            if agent_spec and hasattr(agent_spec, "simulator") and isinstance(agent_spec.simulator, dict):
                val = agent_spec.simulator.get(name, 0.5)
            agent_values.append(float(val))
        agent_tensor = torch.tensor([agent_values], dtype=torch.float, device=device)

        with torch.no_grad():
            eta_min, _ = self._do_predict_efficacy(
                active_tensor, agent_tensor, t_span
            )

        predicted_val = float(eta_min.item())
        return max(0.0, min(1.0, predicted_val))


import typing
if typing.TYPE_CHECKING:
    from runtime.factory import AgentEfficacyModel
_istype_GATAgentEfficacyModel: typing.Type[AgentEfficacyModel] = GATAgentEfficacyModel
