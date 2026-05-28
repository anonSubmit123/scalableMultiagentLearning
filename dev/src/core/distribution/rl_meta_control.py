from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .ic_map import SpatialTemporalICMap
from .risk_prediction import RiskPrediction
from .risk_transformer import SpatialTemporalRiskMap


class RLMetaActorNetwork(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class InterventionDecision:
    degree: float
    max_components: int
    max_agents: int
    selected_components: tuple[str, ...] = ()


@dataclass(frozen=True)
class AbstractObservation:
    total_risk: float
    concentration: float
    growth_pressure: float
    cascade_potential: float
    uncertainty: float
    system_slack: float

    def to_array(self) -> list[float]:
        return [
            self.total_risk,
            self.concentration,
            self.growth_pressure,
            self.cascade_potential,
            self.uncertainty,
            self.system_slack,
        ]


@dataclass
class RLMetaInterventionPolicy:
    low_threshold: float = 0.25
    high_threshold: float = 0.60
    weights: dict[str, float] = field(default_factory=lambda: {
        "total_risk": 0.0,
        "concentration": 1.0,
        "growth_pressure": 0.0,
        "cascade_potential": 0.0,
        "uncertainty": 0.0,
        "system_slack": 0.0,
    })
    model_checkpoint_path: str | None = None

    def compute_observation(
        self,
        ic_map: SpatialTemporalICMap | SpatialTemporalRiskMap,
        prediction: RiskPrediction,
        agent_count: int,
    ) -> AbstractObservation:
        if isinstance(ic_map, SpatialTemporalRiskMap):
            components_list = list(ic_map.components.values())
        else:
            components_list = list(ic_map.components)

        growth_values: list[float] = []
        uncertainty_values: list[float] = []
        for comp in components_list:
            meta = comp.metadata if hasattr(comp, "metadata") and comp.metadata else {}

            growth = meta.get("growth_rate", meta.get("growth", 0.0))
            try:
                growth_values.append(float(growth))
            except (ValueError, TypeError):
                growth_values.append(0.0)

            unc = meta.get("uncertainty", meta.get("predictive_uncertainty", 0.0))
            try:
                uncertainty_values.append(float(unc))
            except (ValueError, TypeError):
                uncertainty_values.append(0.0)

        risk_values = list(prediction.risk_by_component.values()) if prediction.risk_by_component else []
        total_risk = sum(max(val, 0.0) for val in risk_values)

        peak_risk = max(risk_values, default=0.0)
        concentration = peak_risk / total_risk if total_risk > 0.0 else 0.0

        growth_pressure = sum(growth_values) / len(growth_values) if growth_values else 0.0

        cascade_values = list(prediction.cascading_risk.values()) if prediction.cascading_risk else []
        total_cascade = sum(max(val, 0.0) for val in cascade_values)
        cascade_potential = total_cascade / total_risk if total_risk > 0.0 else 0.0

        uncertainty = sum(uncertainty_values) / len(uncertainty_values) if uncertainty_values else 0.0

        system_slack = float(agent_count) / total_risk if total_risk > 0.0 else 1.0

        return AbstractObservation(
            total_risk=total_risk,
            concentration=concentration,
            growth_pressure=growth_pressure,
            cascade_potential=cascade_potential,
            uncertainty=uncertainty,
            system_slack=system_slack,
        )

    def compute_deterministic_degree(self, obs: AbstractObservation) -> float:
        is_pure_concentration = (
            self.weights.get("concentration", 1.0) == 1.0 and
            all(self.weights.get(k, 0.0) == 0.0 for k in [
                "total_risk", "growth_pressure", "cascade_potential", "uncertainty", "system_slack"
            ])
        )

        if is_pure_concentration:
            if obs.concentration >= self.high_threshold:
                return 1.0
            elif obs.concentration >= self.low_threshold:
                return 0.5
            else:
                return 0.25

        threat_score = (
            self.weights.get("total_risk", 0.0) * obs.total_risk +
            self.weights.get("concentration", 1.0) * obs.concentration +
            self.weights.get("growth_pressure", 0.2) * obs.growth_pressure +
            self.weights.get("cascade_potential", 0.5) * obs.cascade_potential +
            self.weights.get("uncertainty", 0.1) * obs.uncertainty +
            self.weights.get("system_slack", -0.1) * obs.system_slack
        )

        if threat_score >= self.high_threshold:
            degree = 1.0
        elif threat_score >= self.low_threshold:
            span = self.high_threshold - self.low_threshold
            fraction = (threat_score - self.low_threshold) / span if span > 0.0 else 0.0
            degree = 0.5 + 0.5 * fraction
        else:
            fraction = threat_score / self.low_threshold if self.low_threshold > 0.0 else 0.0
            degree = 0.25 * fraction

        return max(0.0, min(1.0, degree))

    def select(
        self,
        ic_map: SpatialTemporalICMap | SpatialTemporalRiskMap,
        prediction: RiskPrediction,
        agent_count: int,
    ) -> InterventionDecision:
        obs = self.compute_observation(ic_map, prediction, agent_count)

        degree = None
        checkpoint_path = self.model_checkpoint_path or "rl_meta_model.pt"
        if Path(checkpoint_path).is_file():
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = RLMetaActorNetwork().to(device)
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    obs_tensor = torch.tensor([obs.to_array()], dtype=torch.float32, device=device)
                    degree = float(model(obs_tensor).cpu().item())
            except Exception as err:
                print(f"Warning: Failed checkpoint inference from {checkpoint_path} ({err}). Falling back to parametric threat policy.")

        if degree is None:
            degree = self.compute_deterministic_degree(obs)

        component_count = max(1, int(_component_count(ic_map) * degree))
        reassigned_agents = max(1, int(agent_count * degree))

        return InterventionDecision(
            degree=degree,
            max_components=component_count,
            max_agents=reassigned_agents,
            selected_components=prediction.prioritized_components[:component_count],
        )


def _component_count(ic_map: SpatialTemporalICMap | SpatialTemporalRiskMap) -> int:
    if isinstance(ic_map, SpatialTemporalRiskMap):
        return len(ic_map.component_demands)
    return len(ic_map.components)


def train_policy_from_data(
    dataset_path: str,
    model_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    from core.distribution.risk_transformer import _load_disaster_sequences
    from core.distribution.ic_map import ICMapBuilder
    from core.distribution.risk_prediction import TransformerRiskPredictor

    print(f"Loading past disaster data from: {dataset_path}")
    sequences = _load_disaster_sequences(dataset_path)
    snapshots = [snap for seq in sequences for snap in seq]
    if not snapshots:
        print("Error: No snapshots found in training dataset.")
        return

    print(f"Loaded {len(snapshots)} snapshots. Constructing abstract observations...")
    ic_builder = ICMapBuilder()
    predictor = TransformerRiskPredictor()
    policy = RLMetaInterventionPolicy()

    obs_list: list[list[float]] = []
    target_list: list[float] = []

    for snap in snapshots:
        ic_map = ic_builder.build(snap)
        prediction = predictor.predict(ic_map)
        agent_count = len(snap.agents) if snap.agents else 10

        obs = policy.compute_observation(ic_map, prediction, agent_count)
        target_degree = policy.compute_deterministic_degree(obs)

        obs_list.append(obs.to_array())
        target_list.append(target_degree)

    x_tensor = torch.tensor(obs_list, dtype=torch.float32)
    y_tensor = torch.tensor(target_list, dtype=torch.float32).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RLMetaActorNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    print(f"Beginning RL meta-control policy network pre-training on device: {device}")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        avg_loss = epoch_loss / len(dataset)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | MSE Loss: {avg_loss:.6f}")

    print(f"Writing trained policy weights to: {model_path}")
    torch.save(model.state_dict(), model_path)
    print("RL meta-control policy network pre-training completed successfully!")
