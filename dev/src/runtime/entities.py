from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from core.algorithms.agent_expedition_ppo import AgentStepResult
from core.algorithms.logical_controller import (
    ControllerPush,
    EnvironmentSnapshot,
    PullRequest,
)
from core.algorithms.shared_policy_manager_ppo import PolicyUpdateStats

from .config import EntityEndpoint, EntityKind

if TYPE_CHECKING:
    from .factory import ExperimentRunContext


class RuntimeEntity(Protocol):
    @property
    def entity_id(self) -> str:
        ...

    @property
    def kind(self) -> EntityKind:
        ...

    @property
    def endpoint(self) -> EntityEndpoint:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def wait(self) -> None:
        ...

    def poll_messages(self) -> int:
        ...

    def reset(self, context: ExperimentRunContext | None = None) -> None:
        ...


class LogicalController(RuntimeEntity, Protocol):
    def step(self, snapshot: EnvironmentSnapshot) -> list[ControllerPush]:
        ...


class Agent(RuntimeEntity, Protocol):
    def receive_push(self, push: ControllerPush) -> None:
        ...

    def step(self, force_report: bool = False) -> AgentStepResult:
        ...


class PolicyManager(RuntimeEntity, Protocol):
    def ingest_pull_request(self, request: PullRequest) -> bool:
        ...

    def run_update_round(self) -> PolicyUpdateStats:
        ...
