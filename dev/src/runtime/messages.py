from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MessageKind(str, Enum):
    CONTROLLER_PUSH = "controller_push"
    PULL_REQUEST = "pull_request"
    TRAJECTORY_REPORT = "trajectory_report"
    AGENT_SNAPSHOT = "agent_snapshot"
    POLICY_REQUEST = "policy_request"
    POLICY_UPDATE = "policy_update"
    LIFECYCLE_COMMAND = "lifecycle_command"
    LIFECYCLE_RESPONSE = "lifecycle_response"
    STATUS_UPDATE = "status_update"


@dataclass(frozen=True)
class RuntimeMessage:
    kind: MessageKind
    sender_id: str
    recipient_id: str
    payload: Any


def encode_message(message: RuntimeMessage) -> bytes:
    return pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)


def decode_message(data: bytes) -> RuntimeMessage:
    message = pickle.loads(data)
    if not isinstance(message, RuntimeMessage):
        raise TypeError("UDP datagram did not contain a RuntimeMessage")
    return message
