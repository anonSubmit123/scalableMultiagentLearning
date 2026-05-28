from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.capabilities import CapabilityProfile


class EntityKind(str, Enum):
    LOGICAL_CONTROLLER = "logical_controller"
    AGENT = "agent"
    POLICY_MANAGER = "policy_manager"
    ADMIN = "admin"


@dataclass(frozen=True)
class EntityEndpoint:
    host: str
    port: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityEndpoint":
        return cls(host=str(data["host"]), port=int(data["port"]))

    def as_socket_address(self) -> tuple[str, int]:
        return (self.host, self.port)


@dataclass(frozen=True)
class PhysicalSystemSpec:
    system_id: str
    host: str
    lifecycle_endpoint: EntityEndpoint | None = None
    launch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicalSystemSpec":
        return cls(
            system_id=str(data["id"]),
            host=str(data.get("host", data["id"])),
            lifecycle_endpoint=(
                EntityEndpoint.from_dict(data["lifecycle_endpoint"])
                if data.get("lifecycle_endpoint") is not None
                else None
            ),
            launch=dict(data.get("launch", {})),
        )


@dataclass(frozen=True)
class LogicalControllerSpec:
    controller_id: str
    endpoint: EntityEndpoint
    physical_system_id: str | None = None
    managed_agents: tuple[str, ...] = ()
    launch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogicalControllerSpec":
        return cls(
            controller_id=str(data["id"]),
            endpoint=EntityEndpoint.from_dict(data["endpoint"]),
            physical_system_id=(
                str(data["physical_system_id"])
                if data.get("physical_system_id") is not None
                else None
            ),
            managed_agents=tuple(str(item) for item in data.get("managed_agents", ())),
            launch=dict(data.get("launch", {})),
        )


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    capability: str | None
    capabilities: CapabilityProfile
    endpoint: EntityEndpoint
    controller_id: str
    policy_manager_id: str
    simulator_id: str | None = None
    simulator: dict[str, Any] = field(default_factory=dict)
    physical_system_id: str | None = None
    launch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        capability_profiles: dict[str, CapabilityProfile] | None = None,
    ) -> "AgentSpec":
        capability_name = (
            str(data["capability"]) if data.get("capability") is not None else None
        )
        if capability_name is not None:
            if capability_profiles is None or capability_name not in capability_profiles:
                raise ValueError(f"unknown capability profile: {capability_name}")
            capabilities = capability_profiles[capability_name]
        elif isinstance(data.get("capabilities"), dict):
            capabilities = CapabilityProfile.from_mapping(data["capabilities"])
        else:
            capabilities = CapabilityProfile.from_legacy_tags(
                tuple(str(item) for item in data.get("capabilities", ()))
            )

        return cls(
            agent_id=str(data["id"]),
            capability=capability_name,
            capabilities=capabilities,
            endpoint=EntityEndpoint.from_dict(data["endpoint"]),
            physical_system_id=(
                str(data["physical_system_id"])
                if data.get("physical_system_id") is not None
                else None
            ),
            controller_id=str(data["controller_id"]),
            policy_manager_id=str(data["policy_manager_id"]),
            simulator_id=(
                str(data["simulator_id"])
                if data.get("simulator_id") is not None
                else None
            ),
            simulator=dict(data.get("simulator", {})),
            launch=dict(data.get("launch", {})),
        )


@dataclass(frozen=True)
class PolicyManagerSpec:
    policy_manager_id: str
    endpoint: EntityEndpoint
    physical_system_id: str | None = None
    launch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyManagerSpec":
        return cls(
            policy_manager_id=str(data["id"]),
            endpoint=EntityEndpoint.from_dict(data["endpoint"]),
            physical_system_id=(
                str(data["physical_system_id"])
                if data.get("physical_system_id") is not None
                else None
            ),
            launch=dict(data.get("launch", {})),
        )


@dataclass(frozen=True)
class AdminSpec:
    admin_id: str
    endpoint: EntityEndpoint
    physical_system_id: str | None = None
    launch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdminSpec":
        return cls(
            admin_id=str(data["id"]),
            endpoint=EntityEndpoint.from_dict(data["endpoint"]),
            physical_system_id=(
                str(data["physical_system_id"])
                if data.get("physical_system_id") is not None
                else None
            ),
            launch=dict(data.get("launch", {})),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    controllers: dict[str, LogicalControllerSpec]
    agents: dict[str, AgentSpec]
    policy_managers: dict[str, PolicyManagerSpec]
    admins: dict[str, AdminSpec] = field(default_factory=dict)
    physical_systems: dict[str, PhysicalSystemSpec] = field(default_factory=dict)
    capability_profiles: dict[str, CapabilityProfile] = field(default_factory=dict)
    components: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        capability_profiles = {
            str(name): CapabilityProfile.from_mapping(profile)
            for name, profile in data.get("capabilities", {}).items()
        }
        controllers = {
            spec.controller_id: spec
            for spec in (
                LogicalControllerSpec.from_dict(item)
                for item in data.get("controllers", ())
            )
        }
        agents = {
            spec.agent_id: spec
            for spec in (
                AgentSpec.from_dict(item, capability_profiles)
                for item in data.get("agents", ())
            )
        }
        policy_managers = {
            spec.policy_manager_id: spec
            for spec in (
                PolicyManagerSpec.from_dict(item)
                for item in data.get("policy_managers", ())
            )
        }
        admins = {
            spec.admin_id: spec
            for spec in (
                AdminSpec.from_dict(item)
                for item in data.get("admins", ())
            )
        }
        physical_systems = {
            spec.system_id: spec
            for spec in (
                PhysicalSystemSpec.from_dict(item)
                for item in data.get("physical_systems", ())
            )
        }
        return cls(
            controllers=controllers,
            agents=agents,
            policy_managers=policy_managers,
            admins=admins,
            physical_systems=physical_systems,
            capability_profiles=capability_profiles,
            components=dict(data.get("components", {})),
        )

    def agents_by_physical_system(self) -> dict[str | None, tuple[AgentSpec, ...]]:
        grouped: dict[str | None, list[AgentSpec]] = {}
        for agent in self.agents.values():
            grouped.setdefault(agent.physical_system_id, []).append(agent)
        return {
            system_id: tuple(agents)
            for system_id, agents in grouped.items()
        }

    def entity_ids_by_physical_system(self) -> dict[str | None, tuple[str, ...]]:
        grouped: dict[str | None, list[str]] = {}
        for controller in self.controllers.values():
            grouped.setdefault(controller.physical_system_id, []).append(
                controller.controller_id
            )
        for agent in self.agents.values():
            grouped.setdefault(agent.physical_system_id, []).append(agent.agent_id)
        for manager in self.policy_managers.values():
            grouped.setdefault(manager.physical_system_id, []).append(
                manager.policy_manager_id
            )
        for admin in self.admins.values():
            grouped.setdefault(admin.physical_system_id, []).append(
                admin.admin_id
            )
        return {
            system_id: tuple(sorted(entity_ids))
            for system_id, entity_ids in grouped.items()
        }

    @classmethod
    def from_json_file(cls, path: str | Path) -> "RuntimeConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def controller_for_agent(self, agent_id: str) -> LogicalControllerSpec:
        agent = self.agents[agent_id]
        return self.controllers[agent.controller_id]

    def policy_manager_for_agent(self, agent_id: str) -> PolicyManagerSpec:
        agent = self.agents[agent_id]
        return self.policy_managers[agent.policy_manager_id]

    def physical_system_id_for_entity(self, entity_id: str) -> str | None:
        if entity_id in self.controllers:
            return self.controllers[entity_id].physical_system_id
        if entity_id in self.agents:
            return self.agents[entity_id].physical_system_id
        if entity_id in self.policy_managers:
            return self.policy_managers[entity_id].physical_system_id
        if entity_id in self.admins:
            return self.admins[entity_id].physical_system_id
        raise KeyError(f"unknown runtime entity: {entity_id}")
