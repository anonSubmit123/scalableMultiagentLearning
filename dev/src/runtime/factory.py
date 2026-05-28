from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import importlib
from typing import Any, Protocol

from core.capabilities import CapabilityProfile
from core.algorithms.agent_expedition_ppo import (
    AgentTransport,
    AgentEnvironment,
    AgentExpeditionGuidedPPO,
    PolicyBankClient,
    ExecutionResult,
)
from core.algorithms.logical_controller import (
    AssignmentOptimizer,
    ControllerPushDispatcher,
    LogicalControllerAlgorithm,
    AgentState,
    IndicativeComponent,
)
from core.algorithms.shared_policy_manager_ppo import (
    SharedPolicyBank,
    SharedPolicyManagerPPO,
)

from .config import (
    AgentSpec,
    EntityEndpoint,
    EntityKind,
    LogicalControllerSpec,
    PolicyManagerSpec,
    RuntimeConfig,
)
from .entities import Agent, LogicalController, PolicyManager, RuntimeEntity


class SimulatorUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentRunContext:
    label: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EntityLifecycleStatus:
    entity_id: str
    exists: bool
    running: bool = False
    done: bool = False
    aborted: bool = False
    detail: str = ""


class RuntimeEntityEndpoint(Protocol):
    entity_id: str
    endpoint: EntityEndpoint

    def reset(self, context: ExperimentRunContext | None = None) -> EntityLifecycleStatus:
        ...

    def start(self, context: ExperimentRunContext | None = None) -> EntityLifecycleStatus:
        ...

    def abort(self, context: ExperimentRunContext | None = None) -> EntityLifecycleStatus:
        ...

    def status(self) -> EntityLifecycleStatus:
        ...

    def results(self) -> Any:
        ...


class EntityAccessor(Protocol):
    @property
    def config(self) -> RuntimeConfig:
        ...

    def entity_endpoint(self, entity_id: str) -> RuntimeEntityEndpoint:
        ...

    def entity_endpoints(
        self,
        kind: EntityKind | None = None,
    ) -> tuple[RuntimeEntityEndpoint, ...]:
        ...

    def controller_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        ...

    def agent_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        ...

    def policy_manager_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        ...


class RuntimeSimulator(Protocol):
    def observe(
        self,
        agent_id: str,
        component_id: str,
        params: dict[str, Any],
    ) -> Any:
        ...

    def operate(
        self,
        agent_id: str,
        component_id: str,
        action: Any,
        params: dict[str, Any],
        stats: SimulatorStats | None = None,
    ) -> ExecutionResult:
        ...

    def reset(self, context: ExperimentRunContext | None = None) -> None:
        ...

    def initialize_stats(self, agent_id: str, params: dict[str, Any]) -> SimulatorStats:
        ...


class ResourceStats(Protocol):
    resource_initial: float
    resource_used: float
    resource_available: float


class OperationStats(Protocol):
    initial: float
    performed: float


class SimulatorStats(Protocol):
    def get_resource_types(self) -> list[str]:
        ...

    def get_resource_stats(self, resource_type: str) -> ResourceStats | None:
        ...

    def get_operation_types(self) -> list[str]:
        ...

    def get_operation_stats(self, operation_type: str) -> OperationStats | None:
        ...


@dataclass
class ConcreteResourceStats:
    resource_initial: float = 0.0
    resource_used: float = 0.0
    resource_available: float = 0.0


@dataclass
class ConcreteOperationStats:
    initial: float = 0.0
    performed: float = 0.0


class ConcreteSimulatorStats:
    def __init__(self) -> None:
        self.resources: dict[str, ConcreteResourceStats] = {}
        self.operations: dict[str, ConcreteOperationStats] = {}

    def get_resource_types(self) -> list[str]:
        return sorted(self.resources.keys())

    def get_resource_stats(self, resource_type: str) -> ResourceStats | None:
        return self.resources.get(resource_type)

    def get_operation_types(self) -> list[str]:
        return sorted(self.operations.keys())

    def get_operation_stats(self, operation_type: str) -> OperationStats | None:
        return self.operations.get(operation_type)


@dataclass
class SimulatorAgentEnvironment:
    agent_id: str
    simulator: RuntimeSimulator
    params: dict[str, Any]
    _stats: SimulatorStats | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if hasattr(self.simulator, "initialize_stats"):
            self._stats = self.simulator.initialize_stats(self.agent_id, self.params)

    @property
    def stats(self) -> SimulatorStats | None:
        return self._stats

    def observe(self, component_id: str) -> Any:
        return self.simulator.observe(self.agent_id, component_id, self.params)

    def execute(self, component_id: str, action: Any) -> ExecutionResult:
        return self.simulator.operate(
            self.agent_id,
            component_id,
            action,
            self.params,
            stats=self._stats,
        )

    def reset(self, context: ExperimentRunContext | None = None) -> None:
        reset = self.simulator.reset
        if len(inspect.signature(reset).parameters) == 0:
            reset()
        else:
            reset(context)
        if hasattr(self.simulator, "initialize_stats"):
            self._stats = self.simulator.initialize_stats(self.agent_id, self.params)


class AgentEfficacyModel(Protocol):
    @property
    def ic_type(self) -> int: ...
    @property
    def active_feature_names(self) -> list[str]: ...
    @property
    def agent_feature_names(self) -> list[str]: ...
    @property
    def feature_to_id(self) -> dict[str, int]: ...
    @property
    def agent_capability(self) -> str | None: ...

    def compute_suppression_ratio(
        self,
        active_ic_feature_values: Any,
        agent_feature_values: Any,
        reference_ic_feature_values: Any,
        t_span: Any,
        steps: int = 10,
        epsilon: float = 1e-6,
    ) -> Any: ...

    def predict_efficacy(
        self,
        agent: AgentState,
        component: IndicativeComponent,
        config: Any = None,
    ) -> float: ...


class ICDynamicsModel(Protocol):
    @property
    def ic_type(self) -> int: ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def feature_to_id(self) -> dict[str, int]: ...

    def forward(
        self,
        feature_values: Any,
        t_span: Any,
        steps: int = 10,
    ) -> tuple[Any, Any]: ...

    def get_active_set(
        self,
        feature_values: Any,
        t_span: Any,
        top_k: int = 3,
    ) -> tuple[list[str], Any, Any]: ...

    def get_active_features(self, component: IndicativeComponent) -> list[str]: ...

    def predict_demand(self, component: IndicativeComponent) -> float: ...


class TaskModelProvider(Protocol):
    def get_efficacy_model(self, ictype: int, meta: dict[str, Any] | None = None) -> AgentEfficacyModel:
        ...

    def get_icdynamics_model(self, ictype: int, meta: dict[str, Any] | None = None) -> ICDynamicsModel:
        ...


class RuntimeComponentProvider(Protocol):
    def create_assignment_optimizer(
        self,
        spec: LogicalControllerSpec,
    ) -> AssignmentOptimizer:
        ...

    def create_task_model_provider(self, spec: LogicalControllerSpec) -> TaskModelProvider:
        ...

    def create_agent_policy_bank(self, spec: AgentSpec) -> PolicyBankClient:
        ...

    def create_agent_environment(self, spec: AgentSpec) -> AgentEnvironment:
        ...

    def validate_simulators(self, config: RuntimeConfig) -> None:
        ...

    def create_shared_policy_bank(self, spec: PolicyManagerSpec) -> SharedPolicyBank:
        ...


@dataclass(frozen=True)
class HostEntityLifecycleCommand:
    action: str
    physical_system_id: str
    entity_ids: tuple[str, ...]
    context: ExperimentRunContext | None = None
    reply_endpoint: EntityEndpoint | None = None
    config_payload: dict[str, Any] | None = None
    request_payload: Any | None = None


@dataclass(frozen=True)
class HostEntityLifecycleResponse:
    action: str
    physical_system_id: str
    ok: bool
    entity_status: dict[str, bool]
    message: str = ""
    response_payload: Any | None = None


@dataclass(frozen=True)
class EntityLifecycleCommand:
    action: str
    entity_id: str
    context: ExperimentRunContext | None = None
    reply_endpoint: EntityEndpoint | None = None


@dataclass(frozen=True)
class EntityLifecycleResponse:
    action: str
    entity_id: str
    ok: bool
    status: EntityLifecycleStatus
    message: str = ""
    response_payload: Any | None = None


@dataclass(frozen=True)
class AgentSnapshot:
    agent_id: str
    time: int
    location: Any | None
    activity_id: str | None
    component_observations: dict[str, Any]
    capabilities: CapabilityProfile
    workload: float = 0.0


@dataclass(frozen=True)
class PolicyKey:
    activity_id: str
    capabilities: CapabilityProfile


@dataclass(frozen=True)
class PolicyRequest:
    agent_id: str
    key: PolicyKey
    current_policy_id: str | None = None
    current_version: int | None = None


@dataclass(frozen=True)
class PolicyUpdateNotification:
    key: PolicyKey
    policy_id: str
    version: int
    policy_blob: bytes
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrajectoryUpdate:
    report: Any


class HostEntityLifecycleClient(Protocol):
    def configure_host(
        self,
        physical_system_id: str,
        config_payload: dict[str, Any],
        context: ExperimentRunContext | None = None,
    ) -> HostEntityLifecycleResponse:
        ...

    def ensure_entities(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
        context: ExperimentRunContext | None = None,
    ) -> HostEntityLifecycleResponse:
        ...

    def reset_entities(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
        context: ExperimentRunContext | None = None,
    ) -> HostEntityLifecycleResponse:
        ...

    def stop_entities(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
        context: ExperimentRunContext | None = None,
    ) -> HostEntityLifecycleResponse:
        ...

    def entity_exists(
        self,
        physical_system_id: str,
        entity_id: str,
    ) -> bool:
        ...

    def status_entities(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
    ) -> HostEntityLifecycleResponse:
        ...

    def entity_results(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
    ) -> HostEntityLifecycleResponse:
        ...


def load_component_provider(path: str) -> RuntimeComponentProvider:
    module_name, _, object_name = path.partition(":")
    if not object_name:
        module_name, _, object_name = path.rpartition(".")
    if not module_name or not object_name:
        raise ValueError("provider must be 'module:object' or 'module.object'")

    module = importlib.import_module(module_name)
    obj: Any = getattr(module, object_name)
    return obj() if isinstance(obj, type) else obj


def validate_configured_simulators(
    config: RuntimeConfig,
    provider: RuntimeComponentProvider,
) -> None:
    validator = getattr(provider, "validate_simulators", None)
    if validator is not None:
        validator(config)
        return

    agents_requiring_simulators = [
        spec for spec in config.agents.values() if spec.simulator_id is not None
    ]
    if not agents_requiring_simulators:
        return

    raise SimulatorUnavailableError(
        "runtime config references simulator-backed agents, but provider "
        f"{provider.__class__.__name__} does not implement validate_simulators"
    )


@dataclass(frozen=True)
class DistributedEntitySystem:
    config: RuntimeConfig
    lifecycle_client: HostEntityLifecycleClient
    context: ExperimentRunContext | None = None
    config_payload: dict[str, Any] | None = None
    _configured_hosts: set[str] = field(default_factory=set, init=False, repr=False)

    def _groups(self) -> dict[str, tuple[str, ...]]:
        return self._groups_for(
            tuple(self.config.controllers)
            + tuple(self.config.agents)
            + tuple(self.config.policy_managers)
        )

    def _groups_for(self, entity_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for entity_id in entity_ids:
            system_id = self._physical_system_id_for_entity(entity_id)
            grouped.setdefault(system_id, []).append(entity_id)
        return {
            system_id: tuple(sorted(group_entity_ids))
            for system_id, group_entity_ids in grouped.items()
        }

    def _physical_system_id_for_entity(self, entity_id: str) -> str:
        system_id = self.config.physical_system_id_for_entity(entity_id)
        if system_id is None:
            raise RuntimeError(
                f"entity '{entity_id}' has no physical_system_id for distributed "
                "management"
            )
        return system_id

    def _configure_host(self, system_id: str) -> None:
        if self.config_payload is None or system_id in self._configured_hosts:
            return
        response = self.lifecycle_client.configure_host(
            system_id,
            self.config_payload,
            self.context,
        )
        if not response.ok:
            raise RuntimeError(response.message)
        self._configured_hosts.add(system_id)

    def start(self) -> None:
        for system_id, entity_ids in self._groups().items():
            self._configure_host(system_id)
            response = self.lifecycle_client.ensure_entities(
                system_id,
                entity_ids,
                self.context,
            )
            if not response.ok:
                raise RuntimeError(response.message)

    def stop(self) -> None:
        for system_id, entity_ids in self._groups().items():
            self._configure_host(system_id)
            response = self.lifecycle_client.stop_entities(
                system_id,
                entity_ids,
                self.context,
            )
            if not response.ok:
                raise RuntimeError(response.message)

    def reset(self, context: ExperimentRunContext | None = None) -> None:
        active_context = context or self.context
        for system_id, entity_ids in self._groups().items():
            self._configure_host(system_id)
            response = self.lifecycle_client.reset_entities(
                system_id,
                entity_ids,
                active_context,
            )
            if not response.ok:
                raise RuntimeError(response.message)

    def entity_exists(self, entity_id: str) -> bool:
        system_id = self._physical_system_id_for_entity(entity_id)
        self._configure_host(system_id)
        return self.lifecycle_client.entity_exists(system_id, entity_id)

    def status(self) -> dict[str, EntityLifecycleStatus]:
        statuses: dict[str, EntityLifecycleStatus] = {}
        for system_id, entity_ids in self._groups().items():
            self._configure_host(system_id)
            response = self.lifecycle_client.status_entities(system_id, entity_ids)
            if not response.ok:
                raise RuntimeError(response.message)
            payload = response.response_payload
            if isinstance(payload, dict):
                statuses.update(
                    {
                        entity_id: status
                        for entity_id, status in payload.items()
                        if isinstance(status, EntityLifecycleStatus)
                    }
                )
        return statuses

    def results(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for system_id, entity_ids in self._groups().items():
            self._configure_host(system_id)
            response = self.lifecycle_client.entity_results(system_id, entity_ids)
            if not response.ok:
                raise RuntimeError(response.message)
            if isinstance(response.response_payload, dict):
                results.update(response.response_payload)
        return results


class RuntimeEntityFactory(EntityAccessor, Protocol):
    @property
    def config(self) -> RuntimeConfig:
        ...

    def create_logical_controller(
        self,
        controller_id: str,
    ) -> LogicalController:
        ...

    def create_agent(self, agent_id: str) -> Agent:
        ...

    def create_policy_manager(self, policy_manager_id: str) -> PolicyManager:
        ...

    def create_admin_entity(
        self,
        admin_id: str,
    ) -> RuntimeEntity:
        ...

    def entity_endpoint(self, entity_id: str) -> RuntimeEntityEndpoint:
        ...

    def entity_endpoints(
        self,
        kind: EntityKind | None = None,
    ) -> tuple[RuntimeEntityEndpoint, ...]:
        ...

    def controller_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        ...

    def agent_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        ...

    def policy_manager_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        ...

    def create_distributed_system(
        self,
        context: ExperimentRunContext | None = None,
    ) -> DistributedEntitySystem:
        ...


def build_controller_algorithm(
    spec: LogicalControllerSpec,
    components: RuntimeComponentProvider,
    dispatcher: ControllerPushDispatcher,
) -> LogicalControllerAlgorithm:
    return LogicalControllerAlgorithm(
        optimizer=components.create_assignment_optimizer(spec),
        task_model_provider=components.create_task_model_provider(spec),
        push_dispatcher=dispatcher,
    )


def build_agent_algorithm(
    spec: AgentSpec,
    components: RuntimeComponentProvider,
    transport: AgentTransport,
) -> AgentExpeditionGuidedPPO:
    return AgentExpeditionGuidedPPO(
        agent_id=spec.agent_id,
        capabilities=spec.capabilities,
        policy_bank=components.create_agent_policy_bank(spec),
        environment=components.create_agent_environment(spec),
        transport=transport,
    )


def build_policy_manager_algorithm(
    spec: PolicyManagerSpec,
    components: RuntimeComponentProvider,
) -> SharedPolicyManagerPPO:
    return SharedPolicyManagerPPO(
        policy_bank=components.create_shared_policy_bank(spec),
    )
