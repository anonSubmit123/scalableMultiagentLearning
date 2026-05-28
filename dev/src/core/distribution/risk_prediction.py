from __future__ import annotations

from dataclasses import dataclass

from core.algorithms.logical_controller import ComponentId
from .ic_map import SpatialTemporalICMap
from .risk_transformer import SpatialTemporalRiskMap


@dataclass(frozen=True)
class RiskPrediction:
    risk_by_component: dict[ComponentId, float]
    cascading_risk: dict[ComponentId, float]
    prioritized_components: tuple[ComponentId, ...]


@dataclass
class TransformerRiskPredictor:
    cascade_weight: float = 0.25

    def predict(
        self,
        ic_map: SpatialTemporalICMap | SpatialTemporalRiskMap,
    ) -> RiskPrediction:
        cascading_risk: dict[ComponentId, float] = {}
        risk_by_component: dict[ComponentId, float] = {}

        if isinstance(ic_map, SpatialTemporalRiskMap):
            risk_by_component = dict(ic_map.component_demands)
            cascading_risk = {
                component_id: 0.0
                for component_id in ic_map.component_demands
            }
            return RiskPrediction(
                risk_by_component=risk_by_component,
                cascading_risk=cascading_risk,
                prioritized_components=ic_map.prioritized_components,
            )

        ordered_components = ic_map.components
        for index, component in enumerate(ordered_components):
            component_id = component.component_id
            neighboring_components = (
                ordered_components[max(0, index - 1) : index]
                + ordered_components[index + 1 : index + 2]
            )
            neighbor_demand = sum(neighbor.demand for neighbor in neighboring_components)
            cascade = self.cascade_weight * neighbor_demand
            cascading_risk[component_id] = cascade
            risk_by_component[component_id] = component.demand + cascade

        prioritized = tuple(
            component_id
            for component_id, _ in sorted(
                risk_by_component.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        return RiskPrediction(
            risk_by_component=risk_by_component,
            cascading_risk=cascading_risk,
            prioritized_components=prioritized,
        )
