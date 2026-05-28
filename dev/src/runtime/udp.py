from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from types import TracebackType
from typing import Callable

from core.algorithms.agent_expedition_ppo import AgentTransport, TrajectoryReport
from core.algorithms.logical_controller import ControllerPush, PullRequest

from .config import EntityEndpoint, RuntimeConfig
from .factory import (
    EntityAccessor,
    EntityLifecycleStatus,
    ExperimentRunContext,
    HostEntityLifecycleClient,
    HostEntityLifecycleResponse,
    RuntimeEntityEndpoint,
)
from .messages import MessageKind, RuntimeMessage, decode_message, encode_message
from .transport import RuntimeRealization, RuntimeReceiver, RuntimeSender

LOGGER = logging.getLogger(__name__)

@dataclass
class UdpEntityReceiver(RuntimeReceiver):
    endpoint: EntityEndpoint
    buffer_size: int = 65535
    timeout: float | None = None
    _socket: socket.socket | None = field(default=None, init=False, repr=False)

    def open(self) -> "UdpEntityReceiver":
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(self.endpoint.as_socket_address())
        self._socket.settimeout(self.timeout)
        return self

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def recv_once(self) -> RuntimeMessage:
        if self._socket is None:
            raise RuntimeError("UDP receiver is not open")
        data, sender_address = self._socket.recvfrom(self.buffer_size)
        host, port = self._socket.getsockname()
        LOGGER.debug(f"Received message from {self.endpoint} from {sender_address} to {host}:{port}")
        return decode_message(data)

    def send(self, endpoint: EntityEndpoint, message: RuntimeMessage) -> None:
        if self._socket is None:
            raise RuntimeError("UDP receiver is not open")
        target = endpoint.as_socket_address()
        self._socket.sendto(
            encode_message(message),
            target,
        )
        host, port = self._socket.getsockname()
        LOGGER.debug(f"Sent message to {endpoint} from {host}:{port} to {target[0]}:{target[1]}")

    def serve_forever(self, handler: Callable[[RuntimeMessage], None]) -> None:
        while True:
            try:
                message = self.recv_once()
            except (BlockingIOError, TimeoutError, socket.timeout):
                continue
            except Exception:
                LOGGER.exception(
                    "failed to receive or decode UDP message on %s",
                    self.endpoint,
                )
                continue
            try:
                handler(message)
            except Exception:
                LOGGER.exception(
                    "UDP message handler failed: kind=%s sender=%s recipient=%s payload_type=%s",
                    getattr(message, "kind", None),
                    getattr(message, "sender_id", None),
                    getattr(message, "recipient_id", None),
                    type(getattr(message, "payload", None)).__name__,
                )

    def __enter__(self) -> "UdpEntityReceiver":
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass
class UdpRuntimeSender(RuntimeSender):
    sender_id: str
    _socket: socket.socket = field(
        default_factory=lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
        init=False,
        repr=False,
    )

    def send(
        self,
        endpoint: EntityEndpoint,
        kind: MessageKind,
        recipient_id: str,
        payload: object,
    ) -> None:
        message = RuntimeMessage(
            kind=kind,
            sender_id=self.sender_id,
            recipient_id=recipient_id,
            payload=payload,
        )
        self._socket.sendto(encode_message(message), endpoint.as_socket_address())

    def close(self) -> None:
        self._socket.close()


@dataclass
class UdpEntityEndpoint(RuntimeEntityEndpoint):
    entity_id: str
    endpoint: EntityEndpoint
    physical_system_id: str
    lifecycle_client: HostEntityLifecycleClient

    def reset(
        self,
        context: ExperimentRunContext | None = None,
    ) -> EntityLifecycleStatus:
        response = self.lifecycle_client.reset_entities(
            self.physical_system_id,
            (self.entity_id,),
            context,
        )
        return self._status_from_response(response)

    def start(
        self,
        context: ExperimentRunContext | None = None,
    ) -> EntityLifecycleStatus:
        response = self.lifecycle_client.ensure_entities(
            self.physical_system_id,
            (self.entity_id,),
            context,
        )
        return self._status_from_response(response)

    def abort(
        self,
        context: ExperimentRunContext | None = None,
    ) -> EntityLifecycleStatus:
        response = self.lifecycle_client.stop_entities(
            self.physical_system_id,
            (self.entity_id,),
            context,
        )
        if not response.ok:
            raise RuntimeError(response.message)
        return EntityLifecycleStatus(
            entity_id=self.entity_id,
            exists=response.entity_status.get(self.entity_id, False),
            running=False,
            aborted=True,
        )

    def status(self) -> EntityLifecycleStatus:
        response = self.lifecycle_client.status_entities(
            self.physical_system_id,
            (self.entity_id,),
        )
        return self._status_from_response(response)

    def results(self) -> object:
        response = self.lifecycle_client.entity_results(
            self.physical_system_id,
            (self.entity_id,),
        )
        if not response.ok:
            raise RuntimeError(response.message)
        if isinstance(response.response_payload, dict):
            return response.response_payload.get(self.entity_id)
        return response.response_payload

    def _status_from_response(
        self,
        response: HostEntityLifecycleResponse,
    ) -> EntityLifecycleStatus:
        if not response.ok:
            raise RuntimeError(response.message)
        payload = response.response_payload
        if isinstance(payload, dict) and self.entity_id in payload:
            status = payload[self.entity_id]
            if isinstance(status, EntityLifecycleStatus):
                return status
        return EntityLifecycleStatus(
            entity_id=self.entity_id,
            exists=response.entity_status.get(self.entity_id, False),
            running=response.entity_status.get(self.entity_id, False),
        )


@dataclass
class UdpControllerPushDispatcher:
    controller_id: str
    entities: EntityAccessor
    require_known_agent: bool = True
    sender: RuntimeSender | None = None

    def __post_init__(self) -> None:
        if self.sender is None:
            self.sender = UdpRuntimeSender(self.controller_id)

    def send_push(self, push: ControllerPush) -> None:
        agent_endpoint = self.entities.entity_endpoint(push.agent_id)
        if agent_endpoint is None:
            if self.require_known_agent:
                raise KeyError(f"unknown agent for controller push: {push.agent_id}")
            return
        self.sender.send(
            endpoint=agent_endpoint.endpoint,
            kind=MessageKind.CONTROLLER_PUSH,
            recipient_id=push.agent_id,
            payload=push,
        )

    def close(self) -> None:
        if self.sender is not None:
            self.sender.close()


@dataclass
class UdpAgentTransport(AgentTransport):
    agent_id: str
    config: RuntimeConfig
    sender: RuntimeSender | None = None

    def __post_init__(self) -> None:
        if self.sender is None:
            self.sender = UdpRuntimeSender(self.agent_id)

    def send_pull_request(self, request: PullRequest) -> None:
        controller = self.config.controller_for_agent(self.agent_id)
        if self.sender is None:
            raise RuntimeError("UDP agent transport is not initialized")
        self.sender.send(
            endpoint=controller.endpoint,
            kind=MessageKind.PULL_REQUEST,
            recipient_id=controller.controller_id,
            payload=request,
        )

    def send_trajectory_report(self, report: TrajectoryReport) -> None:
        policy_manager = self.config.policy_manager_for_agent(self.agent_id)
        if self.sender is None:
            raise RuntimeError("UDP agent transport is not initialized")
        self.sender.send(
            endpoint=policy_manager.endpoint,
            kind=MessageKind.TRAJECTORY_REPORT,
            recipient_id=policy_manager.policy_manager_id,
            payload=report,
        )

    def close(self) -> None:
        if self.sender is not None:
            self.sender.close()


@dataclass(frozen=True)
class UdpRuntimeRealization(RuntimeRealization):
    buffer_size: int = 65535
    timeout: float | None = None

    def create_sender(self, sender_id: str) -> RuntimeSender:
        return UdpRuntimeSender(sender_id)

    def create_receiver(self, endpoint: EntityEndpoint) -> RuntimeReceiver:
        return UdpEntityReceiver(
            endpoint=endpoint,
            buffer_size=self.buffer_size,
            timeout=self.timeout,
        )


import typing
if typing.TYPE_CHECKING:
    from runtime.transport import RuntimeReceiver, RuntimeSender, RuntimeRealization
    from runtime.factory import RuntimeEntityEndpoint
    from core.algorithms.logical_controller import ControllerPushDispatcher
    from core.algorithms.agent_expedition_ppo import AgentTransport

_istype_UdpEntityReceiver: typing.Type[RuntimeReceiver] = UdpEntityReceiver
_istype_UdpRuntimeSender: typing.Type[RuntimeSender] = UdpRuntimeSender
_istype_UdpEntityEndpoint: typing.Type[RuntimeEntityEndpoint] = UdpEntityEndpoint
_istype_UdpControllerPushDispatcher: typing.Type[ControllerPushDispatcher] = UdpControllerPushDispatcher
_istype_UdpAgentTransport: typing.Type[AgentTransport] = UdpAgentTransport
_istype_UdpRuntimeRealization: typing.Type[RuntimeRealization] = UdpRuntimeRealization
