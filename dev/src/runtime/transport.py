from __future__ import annotations

from typing import Callable, Protocol

from .config import EntityEndpoint
from .messages import MessageKind, RuntimeMessage


class RuntimeSender(Protocol):
    def send(
        self,
        endpoint: EntityEndpoint,
        kind: MessageKind,
        recipient_id: str,
        payload: object,
    ) -> None:
        ...

    def close(self) -> None:
        ...


class RuntimeReceiver(Protocol):
    def open(self) -> "RuntimeReceiver":
        ...

    def close(self) -> None:
        ...

    def recv_once(self) -> RuntimeMessage:
        ...

    def send(self, endpoint: EntityEndpoint, message: RuntimeMessage) -> None:
        ...

    def serve_forever(self, handler: Callable[[RuntimeMessage], None]) -> None:
        ...

class RuntimeInbox(Protocol):
    def handle(self, message: RuntimeMessage) -> bool:
        ...


class RuntimeRealization(Protocol):
    def create_sender(self, sender_id: str) -> RuntimeSender:
        ...

    def create_receiver(self, endpoint: EntityEndpoint) -> RuntimeReceiver:
        ...
