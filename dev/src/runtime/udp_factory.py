from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Any
import logging
import importlib
import inspect
import json
from pathlib import Path
import tempfile
from subprocess import TimeoutExpired
import time

from .config import AdminSpec, EntityEndpoint, EntityKind, RuntimeConfig
from .factory import (
    DistributedEntitySystem,
    EntityLifecycleCommand,
    EntityLifecycleResponse,
    EntityLifecycleStatus,
    ExperimentRunContext,
    HostEntityLifecycleCommand,
    HostEntityLifecycleResponse,
    RuntimeEntityEndpoint,
    RuntimeSimulator,
    SimulatorAgentEnvironment,
    SimulatorUnavailableError,
    build_agent_algorithm,
    build_controller_algorithm,
    build_policy_manager_algorithm,
)
from .messages import MessageKind, RuntimeMessage, decode_message, encode_message
from .processes import (
    EntityProcessHandle,
    SubprocessEntityProcessLauncher,
    build_launch_command,
)
from .transport import RuntimeRealization
from .udp import (
    UdpAgentTransport,
    UdpControllerPushDispatcher,
    UdpEntityEndpoint,
    UdpRuntimeRealization,
)
from .entities import RuntimeEntity
from .udp_entities import (
    UdpAgentEntity,
    UdpLogicalControllerEntity,
    UdpPolicyManagerEntity,
    UdpAdminEntity,
)

LOGGER = logging.getLogger(__name__)


def _load_object(path: str) -> Any:
    module_name, _, object_name = path.partition(":")
    if not object_name:
        module_name, _, object_name = path.rpartition(".")
    if not module_name or not object_name:
        raise ValueError("component factory must be 'module:object' or 'module.object'")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _accepts_kwarg(callable_object: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return False
    return (
        name in signature.parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    )


def _call_factory(factory: Any, params: dict[str, Any], context: dict[str, Any]) -> Any:
    if not isinstance(factory, type) and not callable(factory):
        return factory

    kwargs = dict(params)
    for name, value in context.items():
        if name not in kwargs and _accepts_kwarg(factory, name):
            kwargs[name] = value
    return factory(**kwargs)


def _has_simulator_methods(candidate: object) -> bool:
    if isinstance(candidate, type):
        return False
    return all(
        callable(getattr(candidate, name, None))
        for name in ("observe", "operate", "reset", "initialize_stats")
    )


def _assert_simulator_protocol(simulator_id: str, simulator: object) -> None:
    missing = [
        name
        for name in ("observe", "operate", "reset", "initialize_stats")
        if not callable(getattr(simulator, name, None))
    ]
    if missing:
        raise SimulatorUnavailableError(
            f"simulator '{simulator_id}' is missing required methods: "
            f"{', '.join(missing)}"
        )


@dataclass
class UdpSocketEntityFactory:
    config: RuntimeConfig
    config_payload: dict[str, Any] | None = None
    realization: RuntimeRealization = field(
        default_factory=lambda: UdpRuntimeRealization(timeout=0.1)
    )
    _component_cache: dict[str, Any] = field(default_factory=dict, init=False)
    _simulators: dict[str, RuntimeSimulator] = field(default_factory=dict, init=False)
    _task_model_provider: Any = field(default=None, init=False)
    _entity_endpoint_cache: dict[str, RuntimeEntityEndpoint] = field(
        default_factory=dict,
        init=False,
    )
    _controller_endpoint_cache: tuple[RuntimeEntityEndpoint, ...] = field(
        default=(),
        init=False,
    )
    _agent_endpoint_cache: tuple[RuntimeEntityEndpoint, ...] = field(
        default=(),
        init=False,
    )
    _policy_manager_endpoint_cache: tuple[RuntimeEntityEndpoint, ...] = field(
        default=(),
        init=False,
    )
    _admin_endpoint_cache: tuple[RuntimeEntityEndpoint, ...] = field(
        default=(),
        init=False,
    )

    def __post_init__(self) -> None:
        self.validate_simulators(self.config)
        self._build_endpoint_cache()

    def create_assignment_optimizer(self, spec: Any) -> Any:
        from .providers import AStarMaxFlowAssignmentOptimizer

        risk_transformer = None
        if "risk_transformer" in self._provider_config():
            risk_transformer = self._create_component(
                "risk_transformer",
                default_factory=None,
                context={"spec": spec, "config": self.config, "factory": self},
                default_scope="singleton",
            )

        intervention_policy = None
        if "rl_meta_policy" in self._provider_config():
            intervention_policy = self._create_component(
                "rl_meta_policy",
                default_factory=None,
                context={"spec": spec, "config": self.config, "factory": self},
                default_scope="singleton",
            )

        return self._create_component(
            "assignment_optimizer",
            default_factory=AStarMaxFlowAssignmentOptimizer,
            context={
                "spec": spec,
                "config": self.config,
                "factory": self,
                "risk_transformer": risk_transformer,
                "intervention_policy": intervention_policy,
            },
        )

    def create_task_model_provider(self, spec: Any) -> Any:
        if self._task_model_provider is None:
            from .providers import GATTaskModelProvider
            self._task_model_provider = self._create_component(
                "task_model_provider",
                default_factory=GATTaskModelProvider,
                context={"spec": spec, "config": self.config, "factory": self},
                default_scope="singleton",
            )
        return self._task_model_provider

    def get_efficacy_model(self, ictype: int, meta: dict[str, Any] | None = None) -> Any:
        if self._task_model_provider is None:
            from .providers import GATTaskModelProvider
            self._task_model_provider = self._create_component(
                "task_model_provider",
                default_factory=GATTaskModelProvider,
                context={"config": self.config, "factory": self},
                default_scope="singleton",
            )
        return self._task_model_provider.get_efficacy_model(ictype, meta)

    def get_icdynamics_model(self, ictype: int, meta: dict[str, Any] | None = None) -> Any:
        if self._task_model_provider is None:
            from .providers import GATTaskModelProvider
            self._task_model_provider = self._create_component(
                "task_model_provider",
                default_factory=GATTaskModelProvider,
                context={"config": self.config, "factory": self},
                default_scope="singleton",
            )
        return self._task_model_provider.get_icdynamics_model(ictype, meta)

    def create_agent_policy_bank(self, spec: Any) -> Any:
        return self._create_component(
            "agent_policy_bank",
            aliases=("policy_bank",),
            default_factory=self._default_policy_bank,
            context={"spec": spec, "config": self.config, "factory": self},
            default_scope="singleton",
        )

    def create_agent_environment(self, spec: Any) -> Any:
        provider = self._provider_config()
        if spec.simulator_id is None and "agent_environment" not in provider:
            raise SimulatorUnavailableError(
                f"agent '{spec.agent_id}' has no simulator_id; configure the "
                "domain simulator this agent should operate against"
            )
        simulator = (
            self._simulators.get(spec.simulator_id)
            if spec.simulator_id is not None
            else None
        )
        if spec.simulator_id is not None and simulator is None:
            raise SimulatorUnavailableError(
                f"simulator '{spec.simulator_id}' for agent '{spec.agent_id}' "
                "was not loaded"
            )
        context = {
            "spec": spec,
            "agent_id": spec.agent_id,
            "params": dict(spec.simulator),
            "config": self.config,
            "factory": self,
        }
        if simulator is not None:
            context["simulator"] = simulator
        return self._create_component(
            "agent_environment",
            default_factory=SimulatorAgentEnvironment,
            context=context,
        )

    def create_shared_policy_bank(self, spec: Any) -> Any:
        return self._create_component(
            "shared_policy_bank",
            aliases=("policy_bank",),
            default_factory=self._default_policy_bank,
            context={"spec": spec, "config": self.config, "factory": self},
            default_scope="singleton",
        )

    def validate_simulators(self, config: RuntimeConfig) -> None:
        self._simulators.clear()
        definitions = self._simulator_definitions(config)
        required = self._required_simulator_agents(config)

        for simulator_id, agents in required.items():
            definition = definitions.get(simulator_id)
            if definition is None:
                agent_list = ", ".join(sorted(agents))
                raise SimulatorUnavailableError(
                    f"simulator '{simulator_id}' is required by agents "
                    f"{agent_list}, but components.simulators has no matching entry"
                )
            self._simulators[simulator_id] = self._load_simulator(
                simulator_id,
                definition,
                agents,
            )

    def _provider_config(self) -> dict[str, Any]:
        provider = self.config.components.get("provider", {})
        if provider is None:
            return {}
        if not isinstance(provider, dict):
            raise TypeError("components.provider must be an object keyed by component")
        return provider

    def _create_component(
        self,
        name: str,
        default_factory: Any,
        context: dict[str, Any],
        aliases: tuple[str, ...] = (),
        default_scope: str = "transient",
    ) -> Any:
        provider = self._provider_config()
        selected_name = name
        raw = provider.get(name)
        for alias in aliases:
            if raw is None and alias in provider:
                selected_name = alias
                raw = provider[alias]
        if raw is None and aliases:
            selected_name = aliases[0]

        if raw is None:
            scope = default_scope
            factory = default_factory
            params: dict[str, Any] = {}
        else:
            selected_name, factory, params, scope = self._component_definition(
                selected_name,
                raw,
            )

        if scope == "singleton" and selected_name in self._component_cache:
            return self._component_cache[selected_name]

        component = _call_factory(factory, params, context)
        if scope == "singleton":
            self._component_cache[selected_name] = component
        return component

    def _component_definition(
        self,
        name: str,
        raw: Any,
    ) -> tuple[str, Any, dict[str, Any], str]:
        if isinstance(raw, str):
            return name, _load_object(raw), {}, "transient"
        if not isinstance(raw, dict):
            return name, raw, {}, "singleton"

        ref = raw.get("ref")
        if ref is not None:
            provider = self._provider_config()
            ref_name = str(ref)
            if ref_name not in provider:
                raise KeyError(f"component '{name}' references unknown '{ref_name}'")
            return self._component_definition(ref_name, provider[ref_name])

        raw_path = raw.get("factory") or raw.get("path") or raw.get("class")
        if raw_path is None:
            raise ValueError(f"component '{name}' is missing factory/path/class")
        scope = str(raw.get("scope", "transient"))
        if scope not in {"transient", "singleton"}:
            raise ValueError(f"component '{name}' has unsupported scope '{scope}'")
        return name, _load_object(str(raw_path)), dict(raw.get("params", {})), scope

    def _default_policy_bank(self) -> Any:
        from .providers import InMemoryPolicyBank

        return InMemoryPolicyBank()

    def _simulator_definitions(self, config: RuntimeConfig) -> dict[str, Any]:
        simulators = config.components.get("simulators", {})
        if not isinstance(simulators, dict):
            raise SimulatorUnavailableError(
                "components.simulators must be an object keyed by simulator id"
            )
        return {str(simulator_id): raw for simulator_id, raw in simulators.items()}

    def _required_simulator_agents(
        self,
        config: RuntimeConfig,
    ) -> dict[str, list[str]]:
        required: dict[str, list[str]] = {}
        for agent in config.agents.values():
            if agent.simulator_id is None:
                continue
            required.setdefault(agent.simulator_id, []).append(agent.agent_id)
        return required

    def _load_simulator(
        self,
        simulator_id: str,
        raw: Any,
        agent_ids: list[str],
    ) -> RuntimeSimulator:
        if isinstance(raw, str):
            factory_path = raw
            params: dict[str, Any] = {}
        elif isinstance(raw, dict):
            raw_path = raw.get("factory") or raw.get("path") or raw.get("class")
            if raw_path is None:
                raise SimulatorUnavailableError(
                    f"simulator '{simulator_id}' is missing factory/path/class"
                )
            factory_path = str(raw_path)
            params = dict(raw.get("params", {}))
        else:
            raise SimulatorUnavailableError(
                f"simulator '{simulator_id}' must be a factory path or object"
            )

        try:
            factory = _load_object(factory_path)
        except (ImportError, AttributeError, ValueError) as exc:
            raise SimulatorUnavailableError(
                f"simulator '{simulator_id}' configured for agents "
                f"{', '.join(sorted(agent_ids))} is not available: cannot load "
                f"'{factory_path}' ({exc})"
            ) from exc

        try:
            simulator = (
                factory
                if _has_simulator_methods(factory)
                else _call_factory(factory, params, {"config": self.config, "factory": self})
            )
        except Exception as exc:
            raise SimulatorUnavailableError(
                f"simulator '{simulator_id}' factory "
                f"'{factory_path}' failed to initialize: {exc}"
            ) from exc

        _assert_simulator_protocol(simulator_id, simulator)
        return simulator

    def create_logical_controller(
        self,
        controller_id: str,
    ) -> UdpLogicalControllerEntity:
        spec = self.config.controllers[controller_id]
        sender = self.realization.create_sender(spec.controller_id)
        dispatcher = UdpControllerPushDispatcher(
            controller_id=spec.controller_id,
            entities=self,
            sender=sender,
        )
        algorithm = build_controller_algorithm(
            spec=spec,
            components=self,
            dispatcher=dispatcher,
        )
        return UdpLogicalControllerEntity(
            receiver=self.realization.create_receiver(spec.endpoint),
            spec=spec,
            algorithm=algorithm,
            config=self.config,
            dispatcher=dispatcher,
        )

    def create_agent(self, agent_id: str) -> UdpAgentEntity:
        spec = self.config.agents[agent_id]
        sender = self.realization.create_sender(spec.agent_id)
        transport = UdpAgentTransport(
            agent_id=spec.agent_id,
            config=self.config,
            sender=sender,
        )
        algorithm = build_agent_algorithm(
            spec=spec,
            components=self,
            transport=transport,
        )
        return UdpAgentEntity(
            receiver=self.realization.create_receiver(spec.endpoint),
            spec=spec,
            algorithm=algorithm,
            transport=transport,
        )

    def create_policy_manager(
        self,
        policy_manager_id: str,
    ) -> UdpPolicyManagerEntity:
        spec = self.config.policy_managers[policy_manager_id]
        algorithm = build_policy_manager_algorithm(
            spec=spec,
            components=self,
        )
        return UdpPolicyManagerEntity(
            receiver=self.realization.create_receiver(spec.endpoint),
            spec=spec,
            algorithm=algorithm,
            config=self.config,
            sender=self.realization.create_sender(spec.policy_manager_id),
        )

    def _build_endpoint_cache(self) -> None:
        self._controller_endpoint_cache = tuple(
            self._create_entity_endpoint(entity_id)
            for entity_id in self.config.controllers
        )
        self._agent_endpoint_cache = tuple(
            self._create_entity_endpoint(entity_id)
            for entity_id in self.config.agents
        )
        self._policy_manager_endpoint_cache = tuple(
            self._create_entity_endpoint(entity_id)
            for entity_id in self.config.policy_managers
        )
        self._admin_endpoint_cache = tuple(
            self._create_entity_endpoint(entity_id)
            for entity_id in self.config.admins
        )
        self._entity_endpoint_cache = {
            endpoint.entity_id: endpoint
            for endpoint in (
                self._controller_endpoint_cache
                + self._agent_endpoint_cache
                + self._policy_manager_endpoint_cache
                + self._admin_endpoint_cache
            )
        }

    def _create_entity_endpoint(self, entity_id: str) -> RuntimeEntityEndpoint:
        if entity_id in self.config.controllers:
            endpoint = self.config.controllers[entity_id].endpoint
        elif entity_id in self.config.agents:
            endpoint = self.config.agents[entity_id].endpoint
        elif entity_id in self.config.policy_managers:
            endpoint = self.config.policy_managers[entity_id].endpoint
        elif entity_id in self.config.admins:
            endpoint = self.config.admins[entity_id].endpoint
        else:
            raise KeyError(f"unknown runtime entity: {entity_id}")
        physical_system_id = self.config.physical_system_id_for_entity(entity_id)
        if physical_system_id is None:
            raise RuntimeError(
                f"entity '{entity_id}' has no physical_system_id for lifecycle "
                "endpoint control"
            )
        return UdpEntityEndpoint(
            entity_id=entity_id,
            endpoint=endpoint,
            physical_system_id=physical_system_id,
            lifecycle_client=UdpHostLifecycleClient(config=self.config),
        )

    def entity_endpoint(self, entity_id: str) -> RuntimeEntityEndpoint:
        try:
            return self._entity_endpoint_cache[entity_id]
        except KeyError as exc:
            raise KeyError(f"unknown runtime entity: {entity_id}") from exc

    def entity_endpoints(
        self,
        kind: EntityKind | None = None,
    ) -> tuple[RuntimeEntityEndpoint, ...]:
        if kind == EntityKind.LOGICAL_CONTROLLER:
            return self.controller_endpoints()
        if kind == EntityKind.AGENT:
            return self.agent_endpoints()
        if kind == EntityKind.POLICY_MANAGER:
            return self.policy_manager_endpoints()
        if kind == EntityKind.ADMIN:
            return self.admin_endpoints()
        if kind is not None:
            raise ValueError(f"unsupported entity kind: {kind}")
        return (
            self._controller_endpoint_cache
            + self._agent_endpoint_cache
            + self._policy_manager_endpoint_cache
            + self._admin_endpoint_cache
        )

    def controller_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        return self._controller_endpoint_cache

    def agent_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        return self._agent_endpoint_cache

    def policy_manager_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        return self._policy_manager_endpoint_cache

    def admin_endpoints(self) -> tuple[RuntimeEntityEndpoint, ...]:
        return self._admin_endpoint_cache

    def create_entity(self, entity_id: str) -> RuntimeEntity:
        if entity_id in self.config.controllers:
            return self.create_logical_controller(entity_id)
        if entity_id in self.config.agents:
            return self.create_agent(entity_id)
        if entity_id in self.config.policy_managers:
            return self.create_policy_manager(entity_id)
        if entity_id in self.config.admins:
            return self.create_admin_entity(entity_id)
        raise KeyError(f"unknown runtime entity: {entity_id}")

    def create_admin_entity(
        self,
        admin_id: str,
    ) -> UdpAdminEntity:
        from .udp import UdpEntityReceiver
        from .udp_entities import UdpAdminEntity

        spec = self.config.admins[admin_id]
        receiver = UdpEntityReceiver(endpoint=spec.endpoint, timeout=0.1)

        return UdpAdminEntity(
            receiver=receiver,
            admin_id=admin_id,
            spec=spec,
            entities=self,
        )

    def create_distributed_system(
        self,
        context: ExperimentRunContext | None = None,
        timeout: float = 5.0,
    ) -> DistributedEntitySystem:
        return DistributedEntitySystem(
            config=self.config,
            lifecycle_client=UdpHostLifecycleClient(
                config=self.config,
                timeout=timeout,
            ),
            context=context,
            config_payload=self.config_payload,
        )


@dataclass(frozen=True)
class UdpHostLifecycleClient:
    config: RuntimeConfig
    timeout: float = 10.0
    sender_id: str = "experiment-lifecycle-client"

    def configure_host(
        self,
        physical_system_id: str,
        config_payload: dict[str, Any],
        context: ExperimentRunContext | None = None,
    ) -> HostEntityLifecycleResponse:
        return self._request(
            "configure",
            physical_system_id,
            (),
            context,
            config_payload=config_payload,
        )

    def ensure_entities(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
        context: ExperimentRunContext | None = None,
    ) -> HostEntityLifecycleResponse:
        return self._request("ensure", physical_system_id, entity_ids, context)

    def reset_entities(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
        context: ExperimentRunContext | None = None,
    ) -> HostEntityLifecycleResponse:
        return self._request("reset", physical_system_id, entity_ids, context)

    def stop_entities(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
        context: ExperimentRunContext | None = None,
    ) -> HostEntityLifecycleResponse:
        return self._request("stop", physical_system_id, entity_ids, context)

    def entity_exists(
        self,
        physical_system_id: str,
        entity_id: str,
    ) -> bool:
        response = self._request("exists", physical_system_id, (entity_id,), None)
        return response.entity_status.get(entity_id, False)

    def status_entities(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
    ) -> HostEntityLifecycleResponse:
        return self._request("status", physical_system_id, entity_ids, None)

    def entity_results(
        self,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
    ) -> HostEntityLifecycleResponse:
        return self._request("results", physical_system_id, entity_ids, None)

    def _request(
        self,
        action: str,
        physical_system_id: str,
        entity_ids: tuple[str, ...],
        context: ExperimentRunContext | None,
        config_payload: dict[str, Any] | None = None,
        request_payload: Any | None = None,
    ) -> HostEntityLifecycleResponse:
        endpoint = self._lifecycle_endpoint(physical_system_id)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("", 0))
            sock.settimeout(self.timeout)
            host, port = sock.getsockname()
            reply_endpoint = EntityEndpoint(host=host, port=port)
            command = HostEntityLifecycleCommand(
                action=action,
                physical_system_id=physical_system_id,
                entity_ids=tuple(sorted(entity_ids)),
                context=context,
                reply_endpoint=reply_endpoint,
                config_payload=config_payload,
                request_payload=request_payload,
            )
            message = RuntimeMessage(
                kind=MessageKind.LIFECYCLE_COMMAND,
                sender_id=self.sender_id,
                recipient_id=physical_system_id,
                payload=command,
            )
            sock.sendto(encode_message(message), endpoint.as_socket_address())
            data, _ = sock.recvfrom(65535)
        response_message = decode_message(data)
        if response_message.kind != MessageKind.LIFECYCLE_RESPONSE:
            raise TypeError("host lifecycle service returned the wrong message kind")
        if not isinstance(response_message.payload, HostEntityLifecycleResponse):
            raise TypeError("host lifecycle response carried the wrong payload")
        return response_message.payload

    def _lifecycle_endpoint(self, physical_system_id: str) -> EntityEndpoint:
        system = self.config.physical_systems.get(physical_system_id)
        if system is None:
            raise KeyError(f"unknown physical system: {physical_system_id}")
        if system.lifecycle_endpoint is None:
            raise RuntimeError(
                f"physical system '{physical_system_id}' has no lifecycle_endpoint"
            )
        return system.lifecycle_endpoint


class UdpHostLifecycleServer:
    buffer_size: int = 65535

    def __init__(self, physical_system_id: str,
        port: int,
        hostifx: str):
        self.physical_system_id = physical_system_id
        self._hostifx = hostifx.strip() if hostifx else ""
        self._port = port
        self.config: RuntimeConfig | None = None
        self._config_path: Path | None = None
        self._process_launcher = SubprocessEntityProcessLauncher()
        self._processes: dict[str, EntityProcessHandle] = {}
        self._socket: socket.socket | None = None
        self._own_endpoint: EntityEndpoint | None = None

    def open(self) -> "UdpHostLifecycleServer":
        target_host = self._hostifx if len(self._hostifx) > 0 else "0.0.0.0"
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((target_host, self._port))
        self._own_endpoint = EntityEndpoint(target_host, self._port)
        LOGGER.info(f"Started lifecycle service at {self._hostifx}:{self._port}")
        return self

    def close(self) -> None:
        self._stop_entities(tuple(self._processes))
        self._remove_config_file()
        if self._socket is not None:
            self._socket.close()
            LOGGER.info(f"Stopped lifecycle service at {self._hostifx}:{self._port}")
            self._socket = None

    def serve_forever(self) -> None:
        if self._socket is None:
            self.open()
        assert self._socket is not None
        while True:
            try:
                data, address = self._socket.recvfrom(self.buffer_size)
                message = decode_message(data)
            except Exception:
                LOGGER.exception(
                    "host lifecycle server failed to receive or decode message on %s:%s",
                    self._hostifx,
                    self._port,
                )
                continue
            try:
                self._handle_message(message, address)
            except Exception:
                LOGGER.exception(
                    "host lifecycle server failed to process message: kind=%s sender=%s recipient=%s payload_type=%s from=%s",
                    getattr(message, "kind", None),
                    getattr(message, "sender_id", None),
                    getattr(message, "recipient_id", None),
                    type(getattr(message, "payload", None)).__name__,
                    address,
                )
                continue

    def _handle_message(
        self,
        message: RuntimeMessage,
        sender_address: tuple[str, int],
    ) -> None:
        if message.kind != MessageKind.LIFECYCLE_COMMAND:
            return
        if not isinstance(message.payload, HostEntityLifecycleCommand):
            raise TypeError("lifecycle command carried the wrong payload")
        command = message.payload
        response = self._handle_command(command)
        if self._socket is not None:
            response_message = RuntimeMessage(
                kind=MessageKind.LIFECYCLE_RESPONSE,
                sender_id=self.physical_system_id,
                recipient_id=message.sender_id,
                payload=response,
            )
            self._socket.sendto(
                encode_message(response_message),
                self._reply_address(command, sender_address),
            )

    def _reply_address(
        self,
        command: HostEntityLifecycleCommand,
        sender_address: tuple[str, int],
    ) -> tuple[str, int]:
        endpoint = command.reply_endpoint
        if endpoint is None or endpoint.host in {"", "0.0.0.0", "::"}:
            return sender_address
        return endpoint.as_socket_address()

    def _handle_command(
        self,
        command: HostEntityLifecycleCommand,
    ) -> HostEntityLifecycleResponse:
        if command.physical_system_id != self.physical_system_id:
            return HostEntityLifecycleResponse(
                action=command.action,
                physical_system_id=self.physical_system_id,
                ok=False,
                entity_status={},
                message=(
                    f"Invalid host: command for {command.physical_system_id} received by "
                    f"{self.physical_system_id}"
                ),
            )
        try:
            response_payload: Any | None = None
            if command.action == "configure":
                self._configure(command.config_payload)
            elif command.action == "ensure":
                response_payload = self._ensure_entities(command.entity_ids)
            elif command.action == "reset":
                response_payload = self._reset_entities(
                    command.entity_ids,
                    command.context,
                )
            elif command.action == "stop":
                response_payload = self._stop_entities(command.entity_ids)
            elif command.action == "exists":
                pass
            elif command.action == "status":
                response_payload = self._entity_status(command.entity_ids)
            elif command.action == "results":
                response_payload = self._entity_results(command.entity_ids)
            else:
                raise ValueError(f"unsupported lifecycle action: {command.action}")
            status = {
                entity_id: self._entity_running(entity_id)
                for entity_id in command.entity_ids
            }
            return HostEntityLifecycleResponse(
                action=command.action,
                physical_system_id=self.physical_system_id,
                ok=True,
                entity_status=status,
                response_payload=response_payload,
            )
        except Exception as exc:
            LOGGER.exception(
                "host lifecycle command failed: action=%s physical_system_id=%s entities=%s",
                command.action,
                command.physical_system_id,
                command.entity_ids,
            )
            status = {
                entity_id: self._entity_running(entity_id)
                for entity_id in command.entity_ids
            }
            return HostEntityLifecycleResponse(
                action=command.action,
                physical_system_id=self.physical_system_id,
                ok=False,
                entity_status=status,
                message=str(exc),
            )

    def _configure(self, config_payload: dict[str, Any] | None) -> None:
        if config_payload is None:
            raise ValueError("configure lifecycle command requires config_payload")
        config = RuntimeConfig.from_dict(config_payload)
        self._stop_entities(tuple(self._processes))
        self._write_config_file(config_payload)
        self.config = config

    def _ensure_entities(
        self,
        entity_ids: tuple[str, ...],
    ) -> dict[str, EntityLifecycleStatus]:
        if self.config is None:
            raise RuntimeError("lifecycle service has not been configured")
        if self._config_path is None:
            raise RuntimeError("lifecycle service has no process config file")
        statuses: dict[str, EntityLifecycleStatus] = {}
        for entity_id in self._lifecycle_start_order(entity_ids):
            self._validate_local_entity(entity_id)
            if self._entity_running(entity_id):
                LOGGER.info(f"Starting pre-existing {entity_id} at {self._hostifx}")
                statuses[entity_id] = self._send_entity_lifecycle(
                    "start",
                    entity_id,
                    None,
                )
                continue
            stale_handle = self._processes.pop(entity_id, None)
            if stale_handle is not None:
                stale_handle.process.wait(timeout=0)
            launch_command = build_launch_command(
                self.config,
                entity_id,
                self._config_path,
            )
            LOGGER.info(f"Starting non-existing {entity_id} at {self._hostifx}")
            self._processes[entity_id] = self._process_launcher.launch(launch_command)
            time.sleep(1)
            statuses[entity_id] = self._send_entity_lifecycle(
                "start",
                entity_id,
                None,
            )
            LOGGER.info(f"Launched {entity_id} at {self._hostifx}")
        return statuses

    def _lifecycle_start_order(
        self,
        entity_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        if self.config is None:
            raise RuntimeError("lifecycle service has not been configured")
        original_order = {
            entity_id: index
            for index, entity_id in enumerate(entity_ids)
        }

        def priority(entity_id: str) -> int:
            if entity_id in self.config.controllers:
                return 0
            if entity_id in self.config.policy_managers:
                return 1
            if entity_id in self.config.agents:
                return 2
            return 3

        return tuple(
            sorted(
                entity_ids,
                key=lambda entity_id: (priority(entity_id), original_order[entity_id]),
            )
        )

    def _write_config_file(self, config_payload: dict[str, Any]) -> None:
        self._remove_config_file()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=f"runtime-{self.physical_system_id}-",
            delete=False,
        ) as config_file:
            json.dump(config_payload, config_file)
            config_file.write("\n")
            self._config_path = Path(config_file.name)

    def _remove_config_file(self) -> None:
        if self._config_path is None:
            return
        try:
            self._config_path.unlink(missing_ok=True)
        finally:
            self._config_path = None

    def _entity_running(self, entity_id: str) -> bool:
        handle = self._processes.get(entity_id)
        return handle is not None and handle.process.poll() is None

    def _process_detail(self, entity_id: str) -> str:
        handle = self._processes.get(entity_id)
        if handle is None:
            return "not launched"
        return_code = handle.process.poll()
        if return_code is None:
            return f"pid={handle.process.pid}"
        return f"exited with code {return_code}"

    def _reset_entities(
        self,
        entity_ids: tuple[str, ...],
        context: ExperimentRunContext | None,
    ) -> dict[str, EntityLifecycleStatus]:
        self._ensure_entities(entity_ids)
        return {
            entity_id: self._send_entity_lifecycle("reset", entity_id, context)
            for entity_id in entity_ids
        }

    def _entity_status(
        self,
        entity_ids: tuple[str, ...],
    ) -> dict[str, EntityLifecycleStatus]:
        statuses: dict[str, EntityLifecycleStatus] = {}
        for entity_id in entity_ids:
            if self._entity_running(entity_id):
                statuses[entity_id] = self._send_entity_lifecycle(
                    "status",
                    entity_id,
                    None,
                )
                continue
            statuses[entity_id] = EntityLifecycleStatus(
                entity_id=entity_id,
                exists=entity_id in self._processes,
                running=False,
                aborted=True,
                detail=self._process_detail(entity_id),
            )
        return statuses

    def _entity_results(self, entity_ids: tuple[str, ...]) -> dict[str, Any]:
        self._ensure_entities(entity_ids)
        results: dict[str, Any] = {}
        for entity_id in entity_ids:
            response = self._request_entity_lifecycle("results", entity_id, None)
            results[entity_id] = response.response_payload
        return results

    def _stop_entities(
        self,
        entity_ids: tuple[str, ...],
    ) -> dict[str, EntityLifecycleStatus]:
        statuses: dict[str, EntityLifecycleStatus] = {}
        for entity_id in entity_ids:
            handle = self._processes.get(entity_id)
            if handle is None:
                statuses[entity_id] = EntityLifecycleStatus(
                    entity_id=entity_id,
                    exists=False,
                    running=False,
                    aborted=True,
                    detail="not launched",
                )
                continue
            if handle.process.poll() is None:
                try:
                    statuses[entity_id] = self._send_entity_lifecycle(
                        "abort",
                        entity_id,
                        None,
                    )
                    handle.process.wait(timeout=1.0)
                    self._processes.pop(entity_id, None)
                except (TimeoutError, TimeoutExpired, OSError):
                    handle.stop()
                    try:
                        handle.process.wait(timeout=1.0)
                    except TimeoutExpired:
                        handle.process.kill()
                        handle.process.wait(timeout=1.0)
                    self._processes.pop(entity_id, None)
                    statuses[entity_id] = EntityLifecycleStatus(
                        entity_id=entity_id,
                        exists=False,
                        running=False,
                        aborted=True,
                        detail="terminated by lifecycle server",
                    )
            else:
                self._processes.pop(entity_id, None)
                statuses[entity_id] = EntityLifecycleStatus(
                    entity_id=entity_id,
                    exists=False,
                    running=False,
                    aborted=True,
                    detail=f"exited with code {handle.process.returncode}",
                )
            LOGGER.info(f"Stopped {entity_id} at {self._hostifx}")
        return statuses

    def _send_entity_lifecycle(
        self,
        action: str,
        entity_id: str,
        context: ExperimentRunContext | None,
    ) -> EntityLifecycleStatus:
        response = self._request_entity_lifecycle(action, entity_id, context)
        if not response.ok:
            raise RuntimeError(response.message)
        return response.status

    def _request_entity_lifecycle(
        self,
        action: str,
        entity_id: str,
        context: ExperimentRunContext | None,
        timeout: float = 10.0,
        attempts: int = 5,
    ) -> EntityLifecycleResponse:
        if self.config is None:
            raise RuntimeError("lifecycle service has not been configured")
        endpoint = self._entity_endpoint(entity_id)
        last_error: TimeoutError | None = None

        for attempt in range(attempts):
            try:
                entity_address = endpoint.as_socket_address()
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.settimeout(timeout)
                    sock.connect(entity_address)
                    host, port = sock.getsockname()
                    command = EntityLifecycleCommand(
                        action=action,
                        entity_id=entity_id,
                        context=context,
                        reply_endpoint=EntityEndpoint(host=host, port=port),
                    )
                    message = RuntimeMessage(
                        kind=MessageKind.LIFECYCLE_COMMAND,
                        sender_id=self.physical_system_id,
                        recipient_id=entity_id,
                        payload=command,
                    )
                    sock.send(encode_message(message))
                    LOGGER.info(f"Sent lifecycle {action} to entity {entity_id} from {host}:{port} to {entity_address[0]}:{entity_address[1]}")
                    data = sock.recv(self.buffer_size)
                response_message = decode_message(data)
                if response_message.kind != MessageKind.LIFECYCLE_RESPONSE:
                    raise TypeError("entity returned the wrong message kind")
                if not isinstance(response_message.payload, EntityLifecycleResponse):
                    raise TypeError("entity lifecycle response carried the wrong payload")
                if response_message.payload.entity_id != entity_id:
                    raise TypeError("entity lifecycle response came from the wrong entity")
                LOGGER.info(f"Received lifecycle response for {action} to entity {entity_id} to {entity_address[0]}:{entity_address[1]}")
                return response_message.payload
            except (socket.timeout, TimeoutError) as exc:
                LOGGER.info(f"Timedout for  {action} response from {entity_id} to {host}:{port} from {entity_address[0]}:{entity_address[1]}")
                last_error = TimeoutError(str(exc))
                if attempt + 1 < attempts:
                    time.sleep(1)
        raise TimeoutError(
            f"entity '{entity_id}' did not respond to lifecycle action '{action}'"
        ) from last_error

    def _entity_endpoint(self, entity_id: str) -> EntityEndpoint:
        if self.config is None:
            raise RuntimeError("lifecycle service has not been configured")
        if entity_id in self.config.controllers:
            return self.config.controllers[entity_id].endpoint
        if entity_id in self.config.agents:
            return self.config.agents[entity_id].endpoint
        if entity_id in self.config.policy_managers:
            return self.config.policy_managers[entity_id].endpoint
        raise KeyError(f"unknown runtime entity: {entity_id}")

    def _validate_local_entity(self, entity_id: str) -> None:
        if self.config is None:
            raise RuntimeError("lifecycle service has not been configured")
        expected_system_id = self.config.physical_system_id_for_entity(entity_id)
        if expected_system_id != self.physical_system_id:
            raise ValueError(
                f"entity '{entity_id}' is configured for physical system "
                f"'{expected_system_id}', not '{self.physical_system_id}'"
            )


import typing
if typing.TYPE_CHECKING:
    from runtime.factory import RuntimeEntityFactory, RuntimeComponentProvider
_istype_UdpSocketEntityFactory_EntityFactory: typing.Type[RuntimeEntityFactory] = UdpSocketEntityFactory
_istype_UdpSocketEntityFactory_ComponentProvider: typing.Type[RuntimeComponentProvider] = UdpSocketEntityFactory
