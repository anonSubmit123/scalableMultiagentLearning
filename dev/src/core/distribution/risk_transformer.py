from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from core.algorithms.logical_controller import (
    ComponentId,
    ComponentLocation,
    ComponentSize,
    EnvironmentSnapshot,
    IndicativeComponent,
)


RISK_TRANSFORMER_WINDOW = 8


@dataclass(frozen=True)
class RiskProgression:
    source_time: int
    window_size: int
    components: dict[ComponentId, IndicativeComponent]

    def to_spatial_temporal_map(self) -> "SpatialTemporalRiskMap":
        return SpatialTemporalRiskMap.from_progression(self)


@dataclass(frozen=True)
class SpatialTemporalRiskMap:
    source_time: int
    window_size: int
    components: dict[ComponentId, IndicativeComponent]
    component_demands: dict[ComponentId, float]
    prioritized_components: tuple[ComponentId, ...]
    features: dict[ComponentId, dict[str, float]]

    @classmethod
    def from_progression(cls, progression: RiskProgression) -> "SpatialTemporalRiskMap":
        component_demands = {
            component_id: component.demand
            for component_id, component in progression.components.items()
        }
        prioritized = tuple(
            component_id
            for component_id, _ in sorted(
                component_demands.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        return cls(
            source_time=progression.source_time,
            window_size=progression.window_size,
            components=dict(progression.components),
            component_demands=component_demands,
            prioritized_components=prioritized,
            features={
                component_id: _component_feature_map(component)
                for component_id, component in progression.components.items()
            },
        )


@dataclass
class RiskTransformerConfig:
    window_size: int = RISK_TRANSFORMER_WINDOW
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.10
    input_features: int = 6
    output_features: int = 5
    max_components: int = 128
    device: str = "cpu"
    model_path: str | None = None
    learning_rate: float = 1.0e-3
    epochs: int = 20
    batch_size: int = 16


class SpatialTemporalRiskTransformer:
    def __init__(self, config: RiskTransformerConfig | None = None) -> None:
        self.config = config or RiskTransformerConfig()
        torch = _load_torch()
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self, cfg: RiskTransformerConfig) -> None:
                super().__init__()
                self.cfg = cfg
                self.input_projection = nn.Linear(cfg.input_features, cfg.d_model)
                self.time_embedding = nn.Embedding(cfg.window_size, cfg.d_model)
                self.component_embedding = nn.Embedding(
                    cfg.max_components,
                    cfg.d_model,
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.nhead,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    layer,
                    num_layers=cfg.num_layers,
                )
                self.output_projection = nn.Linear(
                    cfg.d_model,
                    cfg.output_features,
                )

            def forward(self, tokens: Any, mask: Any) -> Any:
                batch, steps, components, _ = tokens.shape
                encoded = self.input_projection(tokens)
                device = tokens.device
                time_ids = torch.arange(steps, device=device).view(1, steps, 1)
                component_ids = torch.arange(components, device=device).view(
                    1,
                    1,
                    components,
                )
                encoded = (
                    encoded
                    + self.time_embedding(time_ids)
                    + self.component_embedding(component_ids)
                )
                sequence = encoded.reshape(batch, steps * components, self.cfg.d_model)
                padding_mask = ~mask.reshape(batch, steps * components)
                sequence = self.encoder(
                    sequence,
                    src_key_padding_mask=padding_mask,
                )
                latest = sequence.reshape(
                    batch,
                    steps,
                    components,
                    self.cfg.d_model,
                )[:, -1]
                return self.output_projection(latest)

        self._torch = torch
        self._model = _Model(self.config).to(self.config.device)
        self._model.eval()
        if self.config.model_path:
            self.load(self.config.model_path)

    def load(self, path: str | Path) -> None:
        checkpoint = self._torch.load(
            Path(path),
            map_location=self.config.device,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "config": self.config.__dict__,
                "model_state_dict": self._model.state_dict(),
            },
            destination,
        )

    def predict(
        self,
        snapshots: Sequence[EnvironmentSnapshot],
    ) -> RiskProgression:
        if not snapshots:
            return RiskProgression(source_time=0, window_size=0, components={})
        torch = self._torch
        component_ids = _component_ids(snapshots)
        if len(component_ids) > self.config.max_components:
            component_ids = component_ids[: self.config.max_components]
        tokens, mask = _encode_window(
            snapshots=snapshots,
            component_ids=component_ids,
            window_size=self.config.window_size,
        )
        with torch.no_grad():
            prediction = self._model(
                torch.tensor(tokens, dtype=torch.float32, device=self.config.device),
                torch.tensor(mask, dtype=torch.bool, device=self.config.device),
            )[0].detach().cpu().tolist()
        components = _decode_components(
            component_ids=component_ids,
            raw_prediction=prediction,
            fallback_snapshot=snapshots[-1],
        )
        return RiskProgression(
            source_time=snapshots[-1].time,
            window_size=len(snapshots),
            components=components,
        )


@dataclass
class RiskTransformerPipeline:
    config: RiskTransformerConfig = field(default_factory=RiskTransformerConfig)
    model: SpatialTemporalRiskTransformer | None = None
    _torch_unavailable: bool = field(default=False, init=False)

    def predict(
        self,
        snapshots: Sequence[EnvironmentSnapshot],
    ) -> RiskProgression:
        window = tuple(snapshots)[-self.config.window_size :]
        if not window:
            return RiskProgression(source_time=0, window_size=0, components={})
        if self.model is None and not self._torch_unavailable:
            try:
                self.model = SpatialTemporalRiskTransformer(self.config)
            except ModuleNotFoundError:
                self._torch_unavailable = True
        if self.model is not None:
            return self.model.predict(window)
        return _heuristic_progression(window)


def _load_torch() -> Any:
    return importlib.import_module("torch")


def _component_ids(
    snapshots: Sequence[EnvironmentSnapshot],
) -> list[ComponentId]:
    ids: set[ComponentId] = set()
    for snapshot in snapshots:
        ids.update(snapshot.components)
    return sorted(ids)


def _component_vector(component: IndicativeComponent | None) -> list[float]:
    if component is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    location = component.location
    size = component.size
    return [
        float(component.demand),
        0.0 if location is None else float(location.x),
        0.0 if location is None else float(location.y),
        0.0 if size is None else float(size.width),
        0.0 if size is None else float(size.height),
        1.0,
    ]


def _component_feature_map(component: IndicativeComponent) -> dict[str, float]:
    location = component.location
    size = component.size
    return {
        "demand": float(component.demand),
        "location_x": 0.0 if location is None else float(location.x),
        "location_y": 0.0 if location is None else float(location.y),
        "width": 0.0 if size is None else float(size.width),
        "height": 0.0 if size is None else float(size.height),
    }


def _encode_window(
    snapshots: Sequence[EnvironmentSnapshot],
    component_ids: Sequence[ComponentId],
    window_size: int,
) -> tuple[list[list[list[list[float]]]], list[list[list[bool]]]]:
    padded_snapshots: list[EnvironmentSnapshot | None] = [None] * (
        window_size - len(snapshots)
    ) + list(snapshots[-window_size:])
    tokens: list[list[list[float]]] = []
    mask: list[list[bool]] = []
    for snapshot in padded_snapshots:
        step_tokens: list[list[float]] = []
        step_mask: list[bool] = []
        for component_id in component_ids:
            component = None if snapshot is None else snapshot.components.get(component_id)
            step_tokens.append(_component_vector(component))
            step_mask.append(component is not None)
        tokens.append(step_tokens)
        mask.append(step_mask)
    return [tokens], [mask]


def _decode_components(
    component_ids: Sequence[ComponentId],
    raw_prediction: Sequence[Sequence[float]],
    fallback_snapshot: EnvironmentSnapshot,
) -> dict[ComponentId, IndicativeComponent]:
    components: dict[ComponentId, IndicativeComponent] = {}
    for component_id, values in zip(component_ids, raw_prediction):
        fallback = fallback_snapshot.components.get(component_id)
        if fallback is None:
            continue
        demand, x, y, width, height = values
        components[component_id] = IndicativeComponent(
            component_id=component_id,
            demand=max(0.0, float(demand)),
            since=fallback.since,
            ic_type=fallback.ic_type,
            required_capabilities=fallback.required_capabilities,
            location=ComponentLocation(
                x=_finite_or_default(x, fallback.location.x if fallback.location else 0.0),
                y=_finite_or_default(y, fallback.location.y if fallback.location else 0.0),
            ),
            size=ComponentSize(
                width=max(
                    0.0,
                    _finite_or_default(
                        width,
                        fallback.size.width if fallback.size else 0.0,
                    ),
                ),
                height=max(
                    0.0,
                    _finite_or_default(
                        height,
                        fallback.size.height if fallback.size else 0.0,
                    ),
                ),
            ),
            metadata=dict(fallback.metadata),
        )
    return components


def _finite_or_default(value: float, default: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else default


def _heuristic_progression(
    snapshots: Sequence[EnvironmentSnapshot],
) -> RiskProgression:
    latest = snapshots[-1]
    previous = snapshots[-2] if len(snapshots) > 1 else latest
    components: dict[ComponentId, IndicativeComponent] = {}
    for component_id, component in latest.components.items():
        prior = previous.components.get(component_id, component)
        demand_delta = component.demand - prior.demand
        components[component_id] = IndicativeComponent(
            component_id=component_id,
            demand=max(0.0, component.demand + demand_delta),
            since=component.since,
            ic_type=component.ic_type,
            required_capabilities=component.required_capabilities,
            location=component.location,
            size=component.size,
            metadata=dict(component.metadata),
        )
    return RiskProgression(
        source_time=latest.time,
        window_size=len(snapshots),
        components=components,
    )


def _snapshot_from_record(record: dict[str, Any]) -> EnvironmentSnapshot:
    components: dict[ComponentId, IndicativeComponent] = {}
    raw_components = record.get("components", {})
    for component_id, raw in raw_components.items():
        location = raw.get("location") if isinstance(raw, dict) else None
        size = raw.get("size") if isinstance(raw, dict) else None
        components[str(component_id)] = IndicativeComponent(
            component_id=str(component_id),
            demand=float(raw.get("demand", 0.0)),
            since=int(raw.get("since", record.get("time", 0))),
            ic_type=int(raw.get("ic_type", 0)),
            required_capabilities=frozenset(raw.get("required_capabilities", ())),
            location=(
                None
                if not isinstance(location, dict)
                else ComponentLocation(
                    x=float(location.get("x", 0.0)),
                    y=float(location.get("y", 0.0)),
                )
            ),
            size=(
                None
                if not isinstance(size, dict)
                else ComponentSize(
                    width=float(size.get("width", 0.0)),
                    height=float(size.get("height", 0.0)),
                )
            ),
            metadata=dict(raw.get("metadata", {})),
        )
    return EnvironmentSnapshot(
        time=int(record.get("time", 0)),
        components=components,
        agents={},
    )


def _load_disaster_sequences(path: str | Path) -> list[list[EnvironmentSnapshot]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        sequences: list[list[EnvironmentSnapshot]] = []
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            raw = item.get("snapshots") or item.get("sequence") if isinstance(item, dict) else item
            sequences.append([_snapshot_from_record(record) for record in raw])
        return sequences

    data = json.loads(source.read_text())
    raw_sequences = data.get("sequences") if isinstance(data, dict) else data
    if raw_sequences and isinstance(raw_sequences[0], dict):
        raw_sequences = [raw_sequences]
    return [
        [_snapshot_from_record(record) for record in raw_sequence]
        for raw_sequence in raw_sequences
    ]


def _training_examples(
    sequences: Sequence[Sequence[EnvironmentSnapshot]],
    window_size: int,
) -> list[tuple[list[EnvironmentSnapshot], EnvironmentSnapshot]]:
    examples: list[tuple[list[EnvironmentSnapshot], EnvironmentSnapshot]] = []
    for sequence in sequences:
        for target_index in range(1, len(sequence)):
            start = max(0, target_index - window_size)
            examples.append((list(sequence[start:target_index]), sequence[target_index]))
    return examples


def train_from_disaster_data(
    data_path: str | Path,
    output_path: str | Path,
    config: RiskTransformerConfig | None = None,
) -> SpatialTemporalRiskTransformer:
    cfg = config or RiskTransformerConfig()
    cfg.model_path = None
    model = SpatialTemporalRiskTransformer(cfg)
    torch = model._torch
    examples = _training_examples(_load_disaster_sequences(data_path), cfg.window_size)
    if not examples:
        raise ValueError("training data must contain at least one input-target pair")

    optimizer = torch.optim.Adam(model._model.parameters(), lr=cfg.learning_rate)
    loss_fn = torch.nn.MSELoss()
    model._model.train()

    for _ in range(cfg.epochs):
        for batch_start in range(0, len(examples), cfg.batch_size):
            batch = examples[batch_start : batch_start + cfg.batch_size]
            batch_component_ids = sorted(
                {
                    component_id
                    for window, target in batch
                    for snapshot in (*window, target)
                    for component_id in snapshot.components
                }
            )[: cfg.max_components]
            token_batch: list[list[list[list[float]]]] = []
            mask_batch: list[list[list[bool]]] = []
            target_batch: list[list[list[float]]] = []
            target_mask_batch: list[list[bool]] = []
            for window, target in batch:
                tokens, mask = _encode_window(
                    snapshots=window,
                    component_ids=batch_component_ids,
                    window_size=cfg.window_size,
                )
                token_batch.append(tokens[0])
                mask_batch.append(mask[0])
                target_batch.append(
                    [
                        _component_vector(target.components.get(component_id))[: cfg.output_features]
                        for component_id in batch_component_ids
                    ]
                )
                target_mask_batch.append(
                    [
                        component_id in target.components
                        for component_id in batch_component_ids
                    ]
                )

            optimizer.zero_grad()
            prediction = model._model(
                torch.tensor(token_batch, dtype=torch.float32, device=cfg.device),
                torch.tensor(mask_batch, dtype=torch.bool, device=cfg.device),
            )
            target_tensor = torch.tensor(
                target_batch,
                dtype=torch.float32,
                device=cfg.device,
            )
            target_mask = torch.tensor(
                target_mask_batch,
                dtype=torch.bool,
                device=cfg.device,
            ).unsqueeze(-1)
            loss = loss_fn(prediction * target_mask, target_tensor * target_mask)
            loss.backward()
            optimizer.step()

    model._model.eval()
    model.save(output_path)
    return model
