from __future__ import annotations

import subprocess
import sys
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .config import (
    AgentSpec,
    LogicalControllerSpec,
    PolicyManagerSpec,
    RuntimeConfig,
)


@dataclass(frozen=True)
class EntityLaunchCommand:
    entity_id: str
    role: str
    argv: tuple[str, ...]
    physical_system_id: str | None = None


@dataclass
class EntityProcessHandle:
    entity_id: str
    process: subprocess.Popen[bytes]

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()


class EntityProcessLauncher(Protocol):
    def launch(self, command: EntityLaunchCommand) -> EntityProcessHandle:
        ...


@dataclass(frozen=True)
class SubprocessEntityProcessLauncher(EntityProcessLauncher):
    def launch(self, command: EntityLaunchCommand) -> EntityProcessHandle:
        return EntityProcessHandle(
            entity_id=command.entity_id,
            process=subprocess.Popen(command.argv),
        )


def _build_entity_command(
    role: str,
    entity_id: str,
    config_path: str | Path,
    physical_system_id: str | None,
    module: str,
    realization: str,
    launch: dict[str, object],
) -> EntityLaunchCommand:
    if "command" in launch:
        raw = launch["command"]
        if isinstance(raw, str):
            argv = tuple(shlex.split(raw))
        elif isinstance(raw, Sequence):
            argv = tuple(str(item) for item in raw)
        else:
            raise TypeError("launch.command must be a string or sequence")
    else:
        selected_module = str(launch.get("module", module))
        selected_realization = str(launch.get("realization", realization))
        argv = (
            sys.executable,
            "-m",
            selected_module,
            "--role",
            role,
            "--realization",
            selected_realization,
            "--config",
            str(config_path),
            "--entity-id",
            entity_id,
        )
    return EntityLaunchCommand(
        entity_id=entity_id,
        role=role,
        argv=argv,
        physical_system_id=physical_system_id,
    )


def build_launch_commands(
    config: RuntimeConfig,
    config_path: str | Path,
    module: str = "runtime.entity_process",
    realization: str = "udp",
) -> tuple[EntityLaunchCommand, ...]:
    commands: list[EntityLaunchCommand] = []
    for spec in config.controllers.values():
        commands.append(
            _controller_command(spec, config_path, module, realization)
        )
    for spec in config.agents.values():
        commands.append(
            _agent_command(spec, config_path, module, realization)
        )
    for spec in config.policy_managers.values():
        commands.append(
            _policy_manager_command(
                spec,
                config_path,
                module,
                realization,
            )
        )
    return tuple(commands)


def build_launch_command(
    config: RuntimeConfig,
    entity_id: str,
    config_path: str | Path,
    module: str = "runtime.entity_process",
    realization: str = "udp",
) -> EntityLaunchCommand:
    if entity_id in config.controllers:
        return _controller_command(
            config.controllers[entity_id],
            config_path,
            module,
            realization,
        )
    if entity_id in config.agents:
        return _agent_command(
            config.agents[entity_id],
            config_path,
            module,
            realization,
        )
    if entity_id in config.policy_managers:
        return _policy_manager_command(
            config.policy_managers[entity_id],
            config_path,
            module,
            realization,
        )
    raise KeyError(f"unknown runtime entity: {entity_id}")


def _controller_command(
    spec: LogicalControllerSpec,
    config_path: str | Path,
    module: str,
    realization: str,
) -> EntityLaunchCommand:
    return _build_entity_command(
        role="logical_controller",
        entity_id=spec.controller_id,
        config_path=config_path,
        physical_system_id=spec.physical_system_id,
        module=module,
        realization=realization,
        launch=spec.launch,
    )


def _agent_command(
    spec: AgentSpec,
    config_path: str | Path,
    module: str,
    realization: str,
) -> EntityLaunchCommand:
    return _build_entity_command(
        role="agent",
        entity_id=spec.agent_id,
        config_path=config_path,
        physical_system_id=spec.physical_system_id,
        module=module,
        realization=realization,
        launch=spec.launch,
    )


def _policy_manager_command(
    spec: PolicyManagerSpec,
    config_path: str | Path,
    module: str,
    realization: str,
) -> EntityLaunchCommand:
    return _build_entity_command(
        role="policy_manager",
        entity_id=spec.policy_manager_id,
        config_path=config_path,
        physical_system_id=spec.physical_system_id,
        module=module,
        realization=realization,
        launch=spec.launch,
    )


import typing
if typing.TYPE_CHECKING:
    from runtime.processes import EntityProcessLauncher
_istype_SubprocessEntityProcessLauncher: typing.Type[EntityProcessLauncher] = SubprocessEntityProcessLauncher
