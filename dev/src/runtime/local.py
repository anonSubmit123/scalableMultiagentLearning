from __future__ import annotations

from dataclasses import dataclass

from core.algorithms.logical_controller import (
    AgentId,
    ControllerPush,
    ControllerPushDispatcher,
)

from .entities import Agent


@dataclass
class InProcessAgentPushDispatcher(ControllerPushDispatcher):
    agents: dict[AgentId, Agent]
    require_known_agent: bool = True

    def send_push(self, push: ControllerPush) -> None:
        agent = self.agents.get(push.agent_id)
        if agent is None:
            if self.require_known_agent:
                raise KeyError(f"unknown agent for controller push: {push.agent_id}")
            return
        agent.receive_push(push)


import typing
if typing.TYPE_CHECKING:
    from core.algorithms.logical_controller import ControllerPushDispatcher
_istype_InProcessAgentPushDispatcher: typing.Type[ControllerPushDispatcher] = InProcessAgentPushDispatcher
