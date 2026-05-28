from __future__ import annotations

from dataclasses import dataclass

from core.algorithms.logical_controller import (
    ComponentId,
    EnvironmentSnapshot,
    IndicativeComponent,
)


@dataclass(frozen=True)
class SpatialTemporalICMap:
    time: int
    components: tuple[IndicativeComponent, ...]


@dataclass
class ICMapBuilder:
    def build(self, snapshot: EnvironmentSnapshot) -> SpatialTemporalICMap:
        components = tuple(
            sorted(
                snapshot.components.values(),
                key=_location_sort_key,
            )
        )
        return SpatialTemporalICMap(
            time=snapshot.time,
            components=components,
        )


def _with_since(component: IndicativeComponent, since: int) -> IndicativeComponent:
    return IndicativeComponent(
        component_id=component.component_id,
        demand=component.demand,
        since=since,
        ic_type=component.ic_type,
        required_capabilities=component.required_capabilities,
        location=component.location,
        size=component.size,
        metadata=component.metadata,
    )


def _location_sort_key(
    component: IndicativeComponent,
) -> tuple[float, float, ComponentId]:
    location = component.location
    x = float("inf") if location is None else location.x
    y = float("inf") if location is None else location.y
    return x, y, component.component_id
