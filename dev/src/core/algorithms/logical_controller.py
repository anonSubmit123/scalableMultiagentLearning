from __future__ import annotations

from dataclasses import dataclass, field
import os
import torch
from typing import Any, Mapping, Protocol

from core.capabilities import CapabilityProfile


AgentId = str
ActivityId = str
ComponentId = str


@dataclass(frozen=True)
class ComponentLocation:
    x: float
    y: float


@dataclass(frozen=True)
class ComponentSize:
    width: float
    height: float


@dataclass(frozen=True)
class IndicativeComponent:
    component_id: ComponentId
    demand: float
    since: int
    ic_type: int = 0
    required_capabilities: frozenset[str] = frozenset()
    location: ComponentLocation | None = None
    size: ComponentSize | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentState:
    agent_id: AgentId
    activity_id: ActivityId
    workload: float
    capabilities: CapabilityProfile = field(default_factory=CapabilityProfile)


@dataclass(frozen=True)
class Assignment:
    agent_id: AgentId
    activity_id: ActivityId
    component_id: ComponentId


@dataclass(frozen=True)
class PullRequest:
    agent_id: AgentId
    component_id: ComponentId
    efficacy_gap: float
    realized_efficacy: float


@dataclass(frozen=True)
class ControllerPush:
    agent_id: AgentId
    activity_id: ActivityId
    component_id: ComponentId
    expected_efficacy: float
    marginal_gradient: float


@dataclass(frozen=True)
class EnvironmentSnapshot:
    time: int
    components: dict[ComponentId, IndicativeComponent]
    agents: dict[AgentId, AgentState]


@dataclass(frozen=True)
class RedistributionCosts:
    movement: float = 0.0
    switching: float = 0.0
    unresolved_pull: float = 0.0


@dataclass(frozen=True)
class RedistributionWeights:
    movement: float = 0.15
    switching: float = 0.10
    unresolved_pull: float = 0.25


@dataclass(frozen=True)
class RedistributionPlan:
    assignments: dict[AgentId, Assignment]
    global_efficacy: float
    costs: RedistributionCosts = RedistributionCosts()

    def objective_value(self, weights: RedistributionWeights) -> float:
        return (
            self.global_efficacy
            - weights.movement * self.costs.movement
            - weights.switching * self.costs.switching
            - weights.unresolved_pull * self.costs.unresolved_pull
        )


def predict_efficacy_from_provider(
    provider: Any,
    agent: AgentState,
    component: IndicativeComponent,
    config: Any,
) -> float:
    try:
        eff_model = provider.get_efficacy_model(component.ic_type, component.metadata)
        if eff_model is None:
            return 0.5
        return eff_model.predict_efficacy(agent, component, config)
    except Exception as e:
        print(f"Exception during GAT efficacy inference: {e}")
        return 0.5


@dataclass(frozen=True)
class AssignmentContext:
    snapshot: EnvironmentSnapshot
    current_assignment: dict[str, Assignment]
    pull_requests: list[PullRequest]
    task_model_provider: Any
    weights: RedistributionWeights


class AssignmentOptimizer(Protocol):
    def solve(self, context: AssignmentContext) -> RedistributionPlan:
        ...


class ControllerPushDispatcher(Protocol):
    def send_push(self, push: ControllerPush) -> None:
        ...


@dataclass
class LogicalControllerAlgorithm:
    optimizer: AssignmentOptimizer
    task_model_provider: Any
    push_dispatcher: ControllerPushDispatcher | None = None
    controller_interval: int = 25
    pull_pressure_threshold: float = 0.35
    movement_cost_weight: float = 0.15
    switch_cost_weight: float = 0.10
    pull_cost_weight: float = 0.25
    gradient_scale: float = 0.20
    last_update_time: int = 0
    assignment: dict[AgentId, Assignment] = field(default_factory=dict)
    pull_queue: list[PullRequest] = field(default_factory=list)

    def step(
        self,
        snapshot: EnvironmentSnapshot,
        pull_requests: list[PullRequest] | None = None,
    ) -> list[ControllerPush]:
        if pull_requests:
            self.pull_queue.extend(pull_requests)

        pull_pressure = self._pull_pressure(snapshot)
        should_redistribute = (
            snapshot.time - self.last_update_time >= self.controller_interval
            or pull_pressure > self.pull_pressure_threshold
            or not self.assignment
        )
        if not should_redistribute:
            return None

        context = AssignmentContext(
            snapshot=snapshot,
            current_assignment=self.assignment,
            pull_requests=self.pull_queue,
            task_model_provider=self.task_model_provider,
            weights=RedistributionWeights(
                movement=self.movement_cost_weight,
                switching=self.switch_cost_weight,
                unresolved_pull=self.pull_cost_weight,
            ),
        )
        plan = self.optimizer.solve(context)
        mismatch = self._demand_capacity_mismatch(snapshot, plan.assignments)
        pushes = self._build_pushes(snapshot, plan.assignments, mismatch)

        self.assignment = dict(plan.assignments)
        self._clear_resolved_pull_requests()
        self.last_update_time = snapshot.time
        self._deliver_pushes(pushes)
        return pushes

    def _deliver_pushes(self, pushes: list[ControllerPush]) -> None:
        if self.push_dispatcher is None:
            return
        for push in pushes:
            self.push_dispatcher.send_push(push)

    def reset(self) -> None:
        self.last_update_time = 0
        self.assignment.clear()
        self.pull_queue.clear()

    def _pull_pressure(self, snapshot: EnvironmentSnapshot) -> float:
        pressure = 0.0
        for request in self.pull_queue:
            current = self.assignment.get(request.agent_id)
            component_id = current.component_id if current else request.component_id
            demand = snapshot.components.get(
                component_id,
                IndicativeComponent(component_id, 0.0, snapshot.time),
            ).demand
            pressure += demand * request.efficacy_gap
        return pressure

    def _demand_capacity_mismatch(
        self,
        snapshot: EnvironmentSnapshot,
        assignment: dict[AgentId, Assignment],
    ) -> dict[ComponentId, float]:
        served = {component_id: 0.0 for component_id in snapshot.components}
        for item in assignment.values():
            agent = snapshot.agents[item.agent_id]
            component = snapshot.components[item.component_id]
            served[item.component_id] += predict_efficacy_from_provider(
                self.task_model_provider,
                agent,
                component,
                getattr(self, "config", None),
            )
        return {
            component_id: component.demand - served[component_id]
            for component_id, component in snapshot.components.items()
        }

    def _build_pushes(
        self,
        snapshot: EnvironmentSnapshot,
        assignment: dict[AgentId, Assignment],
        mismatch: dict[ComponentId, float],
    ) -> list[ControllerPush]:
        pushes: list[ControllerPush] = []
        for agent_id, item in sorted(assignment.items()):
            agent = snapshot.agents[agent_id]
            component = snapshot.components[item.component_id]
            expected_efficacy = predict_efficacy_from_provider(
                self.task_model_provider,
                agent,
                component,
                getattr(self, "config", None),
            )
            pushes.append(
                ControllerPush(
                    agent_id=agent_id,
                    activity_id=item.activity_id,
                    component_id=item.component_id,
                    expected_efficacy=expected_efficacy,
                    marginal_gradient=self.gradient_scale
                    * mismatch[item.component_id],
                )
            )
        return pushes

    def _clear_resolved_pull_requests(self) -> None:
        assigned = set(self.assignment)
        self.pull_queue = [
            request for request in self.pull_queue if request.agent_id not in assigned
        ]


def estimate_plan_costs(
    next_assignment: dict[AgentId, Assignment],
    current_assignment: dict[AgentId, Assignment],
    pull_requests: list[PullRequest],
) -> RedistributionCosts:
    movement = 0.0
    switching = 0.0
    unresolved_pull = 0.0

    for agent_id, next_item in next_assignment.items():
        current = current_assignment.get(agent_id)
        if current is None:
            continue
        if current.component_id != next_item.component_id:
            movement += 1.0
        if current.activity_id != next_item.activity_id:
            switching += 1.0

    for request in pull_requests:
        next_item = next_assignment.get(request.agent_id)
        if next_item and next_item.component_id == request.component_id:
            unresolved_pull += request.efficacy_gap

    return RedistributionCosts(
        movement=movement,
        switching=switching,
        unresolved_pull=unresolved_pull,
    )
