from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import queue
import socket
import threading
import time
from typing import Any, Mapping

from core.capabilities import capability_key
from core.distribution.risk_transformer import (
    RISK_TRANSFORMER_WINDOW,
    RiskTransformerPipeline,
)
from core.algorithms.agent_expedition_ppo import (
    AgentExpeditionGuidedPPO,
    AgentStepResult,
    TrajectoryReport,
)
from core.algorithms.logical_controller import (
    AgentState,
    ComponentLocation,
    ComponentSize,
    ControllerPush,
    EnvironmentSnapshot,
    IndicativeComponent,
    LogicalControllerAlgorithm,
    PullRequest,
)
from core.algorithms.shared_policy_manager_ppo import (
    PolicyUpdateStats,
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
from .factory import (
    AgentSnapshot,
    EntityAccessor,
    EntityLifecycleCommand,
    EntityLifecycleResponse,
    EntityLifecycleStatus,
    ExperimentRunContext,
    PolicyKey,
    PolicyRequest,
    PolicyUpdateNotification,
    TrajectoryUpdate,
)
from .messages import MessageKind, RuntimeMessage, encode_message
from .transport import RuntimeReceiver, RuntimeSender
from .udp import UdpAgentTransport, UdpControllerPushDispatcher, UdpRuntimeSender


LOGGER = logging.getLogger(__name__)


def _mapping_value(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _number_value(source: Any, name: str) -> float | None:
    value = _mapping_value(source, name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _component_location_from_observation(
    observation: Any,
) -> ComponentLocation | None:
    raw = _mapping_value(observation, "location")
    if isinstance(raw, ComponentLocation):
        return raw
    if raw is None:
        raw = observation
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        x, y = raw[0], raw[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return ComponentLocation(x=float(x), y=float(y))
    x = _number_value(raw, "x")
    y = _number_value(raw, "y")
    if x is None or y is None:
        return None
    return ComponentLocation(x=x, y=y)


def _component_size_from_observation(observation: Any) -> ComponentSize | None:
    raw = _mapping_value(observation, "size")
    if isinstance(raw, ComponentSize):
        return raw
    if raw is None:
        raw = observation
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        width, height = raw[0], raw[1]
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return ComponentSize(width=float(width), height=float(height))
    width = _number_value(raw, "width")
    height = _number_value(raw, "height")
    if width is None or height is None:
        return None
    return ComponentSize(width=width, height=height)


def _component_from_observation(
    component_id: str,
    observation: Any,
) -> IndicativeComponent:
    demand = _number_value(observation, "demand")
    raw_capabilities = _mapping_value(observation, "required_capabilities") or ()
    if isinstance(raw_capabilities, str):
        raw_capabilities = (raw_capabilities,)
    metadata = _mapping_value(observation, "metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return IndicativeComponent(
        component_id=component_id,
        demand=1.0 if demand is None else demand,
        since=int(_number_value(observation, "since") or 0),
        ic_type=int(_number_value(observation, "ic_type") or 0),
        required_capabilities=frozenset(raw_capabilities),
        location=_component_location_from_observation(observation),
        size=_component_size_from_observation(observation),
        metadata=dict(metadata),
    )


def _merge_components(
    current: IndicativeComponent,
    update: IndicativeComponent,
) -> IndicativeComponent:
    return IndicativeComponent(
        component_id=current.component_id,
        demand=max(current.demand, update.demand),
        since=current.since or update.since,
        ic_type=current.ic_type or update.ic_type,
        required_capabilities=(
            current.required_capabilities | update.required_capabilities
        ),
        location=update.location or current.location,
        size=update.size or current.size,
        metadata={**dict(current.metadata), **dict(update.metadata)},
    )


@dataclass
class EntityRuntimeStats:
    ingress_messages_seen: int = 0
    lifecycle_messages_seen: int = 0
    algorithm_iterations: int = 0
    errors: int = 0
    last_error: str = ""
    started_at: float | None = None
    last_reset_at: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "ingress_messages_seen": self.ingress_messages_seen,
            "lifecycle_messages_seen": self.lifecycle_messages_seen,
            "algorithm_iterations": self.algorithm_iterations,
            "errors": self.errors,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "last_reset_at": self.last_reset_at,
        }


@dataclass
class ControllerRuntimeState:
    agent_snapshots: dict[str, AgentSnapshot] = field(default_factory=dict)
    pending_pull_requests: deque[PullRequest] = field(default_factory=deque)
    environment_snapshot: EnvironmentSnapshot | None = None
    risk_progression: object | None = None
    pushes_sent: int = 0
    snapshots_seen: int = 0
    pull_requests_seen: int = 0
    last_push_count: int = 0
    iterations_since_reset: int = 0


@dataclass
class AgentRuntimeState:
    latest_push: ControllerPush | None = None
    active_push: ControllerPush | None = None
    policy_cache_by_key: dict[PolicyKey, PolicyUpdateNotification] = field(
        default_factory=dict
    )
    policy_cache_by_id: dict[str, PolicyUpdateNotification] = field(
        default_factory=dict
    )
    pending_policy_requests: set[PolicyKey] = field(default_factory=set)
    last_snapshot: AgentSnapshot | None = None
    steps: int = 0
    pull_requests_sent: int = 0
    trajectory_reports_sent: int = 0


@dataclass
class PolicyManagerRuntimeState:
    policies_by_key: dict[PolicyKey, PolicyUpdateNotification] = field(
        default_factory=dict
    )
    policies_by_id: dict[str, PolicyUpdateNotification] = field(default_factory=dict)
    trajectory_pools: dict[PolicyKey, list[TrajectoryUpdate]] = field(
        default_factory=dict
    )
    waiters_by_key: dict[PolicyKey, set[str]] = field(default_factory=dict)
    updates_in_flight: set[PolicyKey] = field(default_factory=set)
    policy_requests_seen: int = 0
    trajectory_updates_seen: int = 0
    policy_updates_completed: int = 0
    last_update_stats: PolicyUpdateStats | None = None


@dataclass
class ThreadedUdpEntityBase:
    receiver: RuntimeReceiver
    loop_interval: float = 0.05
    stats: EntityRuntimeStats = field(default_factory=EntityRuntimeStats)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _condition: threading.Condition = field(init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _run_event: threading.Event = field(default_factory=threading.Event, init=False)
    _ingress_thread: threading.Thread | None = field(default=None, init=False)
    _main_thread: threading.Thread | None = field(default=None, init=False)
    _context: ExperimentRunContext | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _subscriptions: list[dict] = field(default_factory=list, init=False)
    _notified_subscriptions: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._state_lock)

    @property
    def entity_id(self) -> str:
        raise NotImplementedError

    @property
    def kind(self) -> EntityKind:
        raise NotImplementedError

    @property
    def endpoint(self) -> EntityEndpoint:
        raise NotImplementedError

    def start(self) -> None:
        LOGGER.info(f"Starting the entity {self.entity_id} at {self.receiver.endpoint.host}:{self.receiver.endpoint.port}")
        with self._state_lock:
            if self._started:
                return
            self._stop_event.clear()
            self._run_event.clear()
            self.receiver.open()
            self.stats.started_at = time.time()
            self._started = True
            self._ingress_thread = threading.Thread(
                target=self._ingress_loop,
                name=f"{self.entity_id}-udp-ingress",
                daemon=True,
            )
            self._main_thread = threading.Thread(
                target=self._main_loop_guarded,
                name=f"{self.entity_id}-algorithm",
                daemon=True,
            )
            self._ingress_thread.start()
            self._main_thread.start()

    def wait(self) -> None:
        with self._condition:
            while not self._stop_event.is_set():
                self._condition.wait()

    def stop(self) -> None:
        self._stop_event.set()
        self._run_event.set()
        with self._condition:
            self._condition.notify_all()
        self.receiver.close()
        current = threading.current_thread()
        for thread in (self._ingress_thread, self._main_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=1.0)
        self._close_role_resources()
        with self._state_lock:
            self._started = False
        LOGGER.info("Stopped the controller")

    def add_subscription(self, sub: dict) -> None:
        with self._condition:
            self._subscriptions.append(sub)

    def remove_subscription(self, token: str) -> None:
        with self._condition:
            self._subscriptions = [s for s in self._subscriptions if s.get("token") != token]
            self._notified_subscriptions.discard(token)

    def reset(self, context: ExperimentRunContext | None = None) -> None:
        with self._condition:
            self._context = context
            self.stats.last_reset_at = time.time()
            self._subscriptions = []
            self._notified_subscriptions = set()
            if context is not None and context.metadata is not None:
                sub = context.metadata.get("implicit_subscription")
                if sub is not None:
                    target_kind = sub.get("subscribe_entity")
                    if target_kind is None or target_kind == self.kind.value:
                        self._subscriptions.append(sub)
            self._reset_role_state(context)
            self._condition.notify_all()
        _log_reset(self.entity_id, context)

    def results(self) -> dict[str, object]:
        with self._state_lock:
            return {
                "entity_id": self.entity_id,
                "kind": self.kind.value,
                "stats": self.stats.as_dict(),
                "role": self._role_results(),
            }

    def _ingress_loop(self) -> None:
        while not self._stop_event.is_set():
            message: RuntimeMessage | None = None
            try:
                message = self.receiver.recv_once()
                self._dispatch_message(message)
            except (BlockingIOError, TimeoutError, socket.timeout):
                continue
            except OSError as exc:
                if not self._stop_event.is_set():
                    self._record_error(exc, message=message, detail="UDP ingress socket error")
                break
            except Exception as exc:
                self._record_error(exc, message=message, detail="UDP message processing failed")

    def _dispatch_message(self, message: RuntimeMessage) -> None:
        LOGGER.info(f"{self.entity_id}: Processing {message.kind} - {type(message).__name__}")
        with self._condition:
            self.stats.ingress_messages_seen += 1
            if message.kind == MessageKind.LIFECYCLE_COMMAND:
                self.stats.lifecycle_messages_seen += 1
                response = self._handle_lifecycle_message(message)
                self._send_lifecycle_response(message, response)
                LOGGER.info(f"{self.entity_id}: Sent response for {message.kind} - {type(response).__name__}")
                if isinstance(message.payload, EntityLifecycleCommand):
                    if message.payload.action == "start" and response.ok:
                        self._run_event.set()
                        self._condition.notify_all()
                    if message.payload.action == "abort":
                        self._stop_event.set()
                        self._run_event.set()
                        self._condition.notify_all()
                return
            accepted = self.handle_request(message)
            if accepted:
                self._condition.notify_all()

    def _handle_lifecycle_message(
        self,
        message: RuntimeMessage,
    ) -> EntityLifecycleResponse:
        if not isinstance(message.payload, EntityLifecycleCommand):
            return self._lifecycle_error(
                "",
                "entity lifecycle message carried the wrong payload",
            )
        command = message.payload
        if command.entity_id != self.entity_id:
            return self._lifecycle_error(command.action, "wrong entity")
        try:
            if command.action == "start":
                LOGGER.info(f"Received start request")
                return self._lifecycle_ok(command.action)
            if command.action == "reset":
                self.reset(command.context)
                return self._lifecycle_ok(command.action)
            if command.action == "status":
                return self._lifecycle_ok(command.action)
            if command.action == "results":
                return self._lifecycle_ok(
                    command.action,
                    response_payload=self.results(),
                )
            if command.action == "abort":
                status = self._status(running=False, aborted=True)
                return EntityLifecycleResponse(
                    action=command.action,
                    entity_id=self.entity_id,
                    ok=True,
                    status=status,
                )
            return self._lifecycle_error(command.action, "unsupported lifecycle action")
        except Exception as exc:
            self._record_error(exc)
            return self._lifecycle_error(command.action, str(exc))

    def _send_lifecycle_response(
        self,
        request: RuntimeMessage,
        response: EntityLifecycleResponse,
    ) -> None:
        command = request.payload
        if not isinstance(command, EntityLifecycleCommand):
            return
        if command.reply_endpoint is None:
            return

        self.receiver.send(
            command.reply_endpoint,
            RuntimeMessage(
                kind=MessageKind.LIFECYCLE_RESPONSE,
                sender_id=self.entity_id,
                recipient_id=request.sender_id,
                payload=response,
            ),
        )

    def _main_loop_guarded(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._run_event.wait(timeout=self.loop_interval):
                    break
            if self._stop_event.is_set():
                return
            self.algorithm_loop()
        except Exception as exc:
            self._record_error(exc)
            self._stop_event.set()
            with self._condition:
                self._condition.notify_all()

    def _status(
        self,
        running: bool | None = None,
        aborted: bool = False,
        detail: str = "",
    ) -> EntityLifecycleStatus:
        return EntityLifecycleStatus(
            entity_id=self.entity_id,
            exists=True,
            running=(not self._stop_event.is_set()) if running is None else running,
            aborted=aborted,
            detail=detail,
        )

    def _lifecycle_ok(
        self,
        action: str,
        response_payload: object | None = None,
    ) -> EntityLifecycleResponse:
        return EntityLifecycleResponse(
            action=action,
            entity_id=self.entity_id,
            ok=True,
            status=self._status(),
            response_payload=response_payload,
        )

    def _lifecycle_error(self, action: str, message: str) -> EntityLifecycleResponse:
        return EntityLifecycleResponse(
            action=action,
            entity_id=self.entity_id,
            ok=False,
            status=self._status(detail=message),
            message=message,
        )

    def _record_error(
        self,
        exc: BaseException,
        message: RuntimeMessage | None = None,
        detail: str = "runtime error",
    ) -> None:
        with self._state_lock:
            self.stats.errors += 1
            self.stats.last_error = str(exc)
        if message is None:
            LOGGER.exception("entity %s %s", self.entity_id, detail)
        else:
            LOGGER.exception(
                "entity %s %s: kind=%s sender=%s recipient=%s payload_type=%s",
                self.entity_id,
                detail,
                message.kind,
                message.sender_id,
                message.recipient_id,
                type(message.payload).__name__,
            )

    def handle_request(self, message: RuntimeMessage) -> bool:
        raise NotImplementedError

    def algorithm_loop(self) -> None:
        raise NotImplementedError

    def _reset_role_state(self, context: ExperimentRunContext | None) -> None:
        raise NotImplementedError

    def _role_results(self) -> dict[str, object]:
        raise NotImplementedError

    def _close_role_resources(self) -> None:
        pass


@dataclass
class UdpLogicalControllerEntity(ThreadedUdpEntityBase):
    spec: LogicalControllerSpec = field(default=None)
    algorithm: LogicalControllerAlgorithm = field(default=None)
    config: RuntimeConfig = field(default=None)
    dispatcher: UdpControllerPushDispatcher = field(default=None)
    state: ControllerRuntimeState = field(default_factory=ControllerRuntimeState)
    risk_transformer: RiskTransformerPipeline = field(
        default_factory=RiskTransformerPipeline,
    )
    _risk_queue: queue.Queue[tuple[EnvironmentSnapshot, ...]] = field(
        default_factory=lambda: queue.Queue(maxsize=1),
        init=False,
    )
    _risk_snapshot_window: deque[EnvironmentSnapshot] = field(
        default_factory=lambda: deque(maxlen=RISK_TRANSFORMER_WINDOW),
        init=False,
    )
    _risk_thread: threading.Thread | None = field(default=None, init=False)

    @property
    def entity_id(self) -> str:
        return self.spec.controller_id

    @property
    def kind(self) -> EntityKind:
        return EntityKind.LOGICAL_CONTROLLER

    @property
    def endpoint(self) -> EntityEndpoint:
        return self.spec.endpoint

    def handle_request(self, message: RuntimeMessage) -> bool:
        if message.kind == MessageKind.PULL_REQUEST:
            if not isinstance(message.payload, PullRequest):
                raise TypeError("pull-request UDP message carried the wrong payload")
            self.state.pending_pull_requests.append(message.payload)
            self.state.pull_requests_seen += 1
            return True
        if message.kind == MessageKind.AGENT_SNAPSHOT:
            if not isinstance(message.payload, AgentSnapshot):
                raise TypeError("agent-snapshot UDP message carried the wrong payload")
            self.state.agent_snapshots[message.payload.agent_id] = message.payload
            self.state.snapshots_seen += 1
            return True
        return False

    def algorithm_loop(self) -> None:
        self._launch_risk_worker()
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait(timeout=self.loop_interval)
                agent_snapshots = tuple(self.state.agent_snapshots.values())
                pull_requests = list(self.state.pending_pull_requests)
                self.state.pending_pull_requests.clear()
            snapshot_start = time.perf_counter()
            snapshot = self._build_environment_snapshot(agent_snapshots)
            snapshot_elapsed = time.perf_counter() - snapshot_start
            if snapshot is None:
                continue

            risk_submit_elapsed = 0.0
            algorithm_step_elapsed = 0.0
            subscription_elapsed = 0.0
            pushes: list[ControllerPush] | None = None
            if snapshot.components:
                risk_start = time.perf_counter()
                self._submit_risk_snapshot(snapshot)
                risk_submit_elapsed = time.perf_counter() - risk_start
                algorithm_start = time.perf_counter()
                pushes = self.algorithm.step(snapshot, pull_requests=pull_requests)
                algorithm_step_elapsed = time.perf_counter() - algorithm_start

            with self._condition:
                self.stats.algorithm_iterations += 1
                self.state.iterations_since_reset += 1
                self.state.environment_snapshot = snapshot
                if pushes is None:
                    self.state.last_push_count = 0
                else:
                    self.state.last_push_count = len(pushes)
                    self.state.pushes_sent += len(pushes)

            subscription_start = time.perf_counter()
            self._evaluate_subscriptions_and_notify(snapshot)
            subscription_elapsed = time.perf_counter() - subscription_start

    def _evaluate_subscriptions_and_notify(self, snapshot: EnvironmentSnapshot) -> None:
        subscriptions = getattr(self, "_subscriptions", None)
        if not subscriptions or snapshot is None:
            return
        if not snapshot.components:
            return

        if getattr(self, "_initial_demand", None) is None:
            self._initial_demand = sum(ic.demand for ic in snapshot.components.values())

        if getattr(self, "_initial_components", None) is None:
            self._initial_components = list(snapshot.components.keys())
            self._initial_ic_count = len(self._initial_components)

        current_demand = sum(ic.demand for ic in snapshot.components.values())
        notified_subs = getattr(self, "_notified_subscriptions", set())

        current_components = snapshot.components
        contained_count = 0
        initial_ic_count = getattr(self, "_initial_ic_count", 0)
        initial_components = getattr(self, "_initial_components", None)

        if initial_components and initial_ic_count > 0:
            for comp_id in initial_components:
                if comp_id not in current_components:
                    contained_count += 1
                elif current_components[comp_id].demand <= 0.05:
                    contained_count += 1

        remaining_count = initial_ic_count - contained_count
        percent_contained = (contained_count / initial_ic_count * 100.0) if initial_ic_count > 0 else 0.0
        percent_remaining = (remaining_count / initial_ic_count * 100.0) if initial_ic_count > 0 else 0.0

        for index, sub in enumerate(subscriptions):
            token = sub.get("token")
            if token in notified_subs:
                continue

            cond_type = sub.get("condition_type")
            threshold = sub.get("threshold", 0.0)
            subscriber_id = sub.get("subscriber_id")
            repeat_count = sub.get("repeat_count", 3)

            if not subscriber_id:
                continue

            if subscriber_id not in self.config.admins and not sub.get("subscriber_endpoint"):
                continue

            triggered = False
            if cond_type == "demand_reduction_percent":
                target_demand = self._initial_demand * (1.0 - threshold / 100.0)
                if current_demand <= target_demand:
                    triggered = True
            elif cond_type == "demand_below_absolute":
                if current_demand <= threshold:
                    triggered = True
            elif cond_type == "assignment_iterations":
                max_iteration = sub.get("iterations", 0.0)
                if self.state.iterations_since_reset > max_iteration:
                    triggered = True
            elif cond_type == "ic_contained_percent":
                if percent_contained >= threshold:
                    triggered = True
            elif cond_type == "ic_remaining_percent":
                if percent_remaining <= threshold:
                    triggered = True

            if triggered:
                if getattr(self, "_containment_reached_time", None) is None:
                    self._containment_reached_time = time.time()
                if not hasattr(self, "_repeat_countdowns"):
                    self._repeat_countdowns = {}
                if token not in self._repeat_countdowns:
                    self._repeat_countdowns[token] = repeat_count
                    notified_subs.add(token)

        if hasattr(self, "_repeat_countdowns"):
            for token, remaining in list(self._repeat_countdowns.items()):
                if remaining <= 0:
                    continue

                target_sub = None
                for sub in subscriptions:
                    if sub.get("token") == token:
                        target_sub = sub
                        break

                if target_sub is None:
                    continue

                subscriber_id = target_sub.get("subscriber_id")

                admin_spec = self.config.admins.get(subscriber_id)
                if admin_spec is not None:
                    sub_endpoint = admin_spec.endpoint
                elif target_sub.get("subscriber_endpoint") is not None:
                    endpoint_dict = target_sub.get("subscriber_endpoint")
                    sub_endpoint = EntityEndpoint(
                        host=endpoint_dict.get("host", "127.0.0.1"),
                        port=int(endpoint_dict.get("port", 0))
                    )
                else:
                    continue

                try:
                    self.dispatcher.sender.send(
                        endpoint=sub_endpoint,
                        kind=MessageKind.STATUS_UPDATE,
                        recipient_id=subscriber_id,
                        payload={
                            "status": "triggered",
                            "condition_type": target_sub.get("condition_type"),
                            "current_demand": current_demand,
                            "token": f"{token}-{self.entity_id}",
                            "iterations": self.state.iterations_since_reset,
                            "percent_contained": percent_contained,
                            "percent_remaining": percent_remaining,
                        }
                    )
                except Exception as e:
                    self._record_error(e)

                self._repeat_countdowns[token] -= 1

    def _launch_risk_worker(self) -> None:
        if self._risk_thread is not None and self._risk_thread.is_alive():
            return
        self._risk_thread = threading.Thread(
            target=self._risk_worker_loop,
            name=f"{self.entity_id}-risk-inference-worker",
            daemon=True,
        )
        self._risk_thread.start()

    def _submit_risk_snapshot(self, snapshot: EnvironmentSnapshot) -> None:
        self._risk_snapshot_window.append(snapshot)
        window = tuple(self._risk_snapshot_window)
        while True:
            try:
                self._risk_queue.get_nowait()
                self._risk_queue.task_done()
            except queue.Empty:
                break
        try:
            self._risk_queue.put_nowait(window)
        except queue.Full:
            pass

    def _risk_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snapshots = self._risk_queue.get(timeout=self.loop_interval)
            except queue.Empty:
                continue
            try:
                risk_model = getattr(self.algorithm, "risk_model", None)
                if risk_model is not None and hasattr(risk_model, "predict"):
                    risk = risk_model.predict(snapshots)
                else:
                    risk = self.risk_transformer.predict(snapshots)
                with self._condition:
                    self.state.risk_progression = risk
            except Exception as exc:
                self._record_error(exc)
            finally:
                self._risk_queue.task_done()

    def step(self, snapshot: EnvironmentSnapshot) -> list[ControllerPush]:
        with self._condition:
            pull_requests = list(self.state.pending_pull_requests)
            self.state.pending_pull_requests.clear()
            self.state.environment_snapshot = snapshot
        return self.algorithm.step(snapshot, pull_requests=pull_requests)

    def _build_environment_snapshot(
        self,
        agent_snapshots: tuple[AgentSnapshot, ...],
    ) -> EnvironmentSnapshot | None:
        if not agent_snapshots:
            return None
        components: dict[str, IndicativeComponent] = {}
        agents: dict[str, AgentState] = {}
        snapshot_time = 0
        for item in agent_snapshots:
            snapshot_time = max(snapshot_time, item.time)
            activity_id = item.activity_id or "unassigned"
            agents[item.agent_id] = AgentState(
                agent_id=item.agent_id,
                activity_id=activity_id,
                workload=item.workload,
                capabilities=item.capabilities,
            )
            for component_id, observation in item.component_observations.items():
                component = _component_from_observation(component_id, observation)
                existing = components.get(component_id)
                components[component_id] = (
                    component
                    if existing is None
                    else _merge_components(existing, component)
                )
        return EnvironmentSnapshot(
            time=snapshot_time,
            components=components,
            agents=agents,
        )

    def reset(self, context: ExperimentRunContext | None = None) -> None:
        super().reset(context)
        self.state.iterations_since_reset = 0
        self.algorithm.reset()

    def _reset_role_state(self, context: ExperimentRunContext | None) -> None:
        self.state.pending_pull_requests.clear()
        self.state.agent_snapshots.clear()
        self.state.environment_snapshot = None
        self.state.risk_progression = None
        self.state.last_push_count = 0
        self._risk_snapshot_window.clear()
        self._initial_demand = None
        self._initial_components = None
        self._initial_ic_count = None
        self._containment_start_time = time.time()
        self._containment_reached_time = None

        while True:
            try:
                self._risk_queue.get_nowait()
                self._risk_queue.task_done()
            except queue.Empty:
                break

    def _role_results(self) -> dict[str, object]:
        ic_demands = {}
        initial_components = getattr(self, "_initial_components", None)
        initial_ic_count = getattr(self, "_initial_ic_count", None) or 0
        percent_contained_current = 0.0
        percent_remaining_current = 0.0

        if initial_components and initial_ic_count > 0:
            current_components = {}
            if self.state.environment_snapshot is not None:
                current_components = self.state.environment_snapshot.components

            contained_count = 0
            for comp_id in initial_components:
                if comp_id not in current_components:
                    contained_count += 1
                elif current_components[comp_id].demand <= 0.05:
                    contained_count += 1
            remaining_count = initial_ic_count - contained_count
            percent_contained_current = (contained_count / initial_ic_count * 100.0)
            percent_remaining_current = (remaining_count / initial_ic_count * 100.0)

        with self._condition:
            if self.state.environment_snapshot is not None:
                for ic_id, ic in self.state.environment_snapshot.components.items():
                    ic_demands[ic_id] = ic.demand
        duration = 0.0
        if getattr(self, "_containment_start_time", None) is not None:
            if getattr(self, "_containment_reached_time", None) is not None:
                duration = self._containment_reached_time - self._containment_start_time
            else:
                duration = time.time() - self._containment_start_time
        return {
            "snapshots_seen": self.state.snapshots_seen,
            "pull_requests_seen": self.state.pull_requests_seen,
            "pushes_sent": self.state.pushes_sent,
            "last_push_count": self.state.last_push_count,
            "known_agents": sorted(self.state.agent_snapshots),
            "risk_window_size": len(self._risk_snapshot_window),
            "ic_demands": ic_demands,
            "containment_time": duration,
            "initial_ic_count": initial_ic_count,
            "percent_contained": percent_contained_current,
            "percent_remaining": percent_remaining_current,
        }

    def _close_role_resources(self) -> None:
        if self._risk_thread is not None and self._risk_thread.is_alive():
            self._risk_thread.join(timeout=1.0)
        self.dispatcher.close()


@dataclass
class UdpAgentEntity(ThreadedUdpEntityBase):
    spec: AgentSpec = field(default=None)
    algorithm: AgentExpeditionGuidedPPO = field(default=None)
    transport: UdpAgentTransport = field(default=None)
    snapshot_interval: float = 1.0
    state: AgentRuntimeState = field(default_factory=AgentRuntimeState)
    _last_snapshot_sent: float = field(default=0.0, init=False)

    @property
    def entity_id(self) -> str:
        return self.spec.agent_id

    @property
    def kind(self) -> EntityKind:
        return EntityKind.AGENT

    @property
    def endpoint(self) -> EntityEndpoint:
        return self.spec.endpoint

    def handle_request(self, message: RuntimeMessage) -> bool:
        if message.kind == MessageKind.CONTROLLER_PUSH:
            if not isinstance(message.payload, ControllerPush):
                raise TypeError("controller-push UDP message carried the wrong payload")
            self.state.latest_push = message.payload
            return True
        if message.kind == MessageKind.POLICY_UPDATE:
            if not isinstance(message.payload, PolicyUpdateNotification):
                raise TypeError("policy-update UDP message carried the wrong payload")
            update = message.payload
            self.state.policy_cache_by_key[update.key] = update
            self.state.policy_cache_by_id[update.policy_id] = update
            self.state.pending_policy_requests.discard(update.key)
            return True
        return False

    def algorithm_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                push = self.state.latest_push
                self.state.latest_push = None
            if push is not None:
                self._activate_push(push)
            result = self.algorithm.step()
            flow_rate = self.spec.simulator.get("flow_rate", 1.0) if self.spec.simulator else 1.0
            if not hasattr(self, "_primary_resource_used"):
                self._primary_resource_used = 0.0
            self._primary_resource_used += flow_rate
            self._after_step(result)
            time.sleep(self.loop_interval)

    def receive_push(self, push: ControllerPush) -> None:
        with self._condition:
            self.state.latest_push = push
            self._condition.notify_all()

    def step(self, force_report: bool = False) -> AgentStepResult:
        result = self.algorithm.step(force_report=force_report)
        self._after_step(result)
        return result

    def _activate_push(self, push: ControllerPush) -> None:
        key = PolicyKey(
            activity_id=push.activity_id,
            capabilities=self.spec.capabilities,
        )
        with self._condition:
            self.state.active_push = push
            if key not in self.state.policy_cache_by_key:
                self._request_policy_once_locked(key)
        self.algorithm.receive_push(push)

    def _request_policy_once_locked(self, key: PolicyKey) -> None:
        if key in self.state.pending_policy_requests:
            return
        self.state.pending_policy_requests.add(key)
        if self.transport.sender is None:
            raise RuntimeError("agent transport sender is not initialized")
        policy_manager = self.transport.config.policy_manager_for_agent(self.entity_id)
        self.transport.sender.send(
            endpoint=policy_manager.endpoint,
            kind=MessageKind.POLICY_REQUEST,
            recipient_id=policy_manager.policy_manager_id,
            payload=PolicyRequest(agent_id=self.entity_id, key=key),
        )

    def _after_step(self, result: AgentStepResult) -> None:
        with self._condition:
            self.stats.algorithm_iterations += 1
            self.state.steps += 1
            if result.pull_requested:
                self.state.pull_requests_sent += 1
            if result.reported:
                self.state.trajectory_reports_sent += 1
            snapshot = self._build_agent_snapshot_locked()
            self.state.last_snapshot = snapshot
        now = time.time()
        if now - self._last_snapshot_sent >= self.snapshot_interval:
            self._last_snapshot_sent = now
            self._send_agent_snapshot(snapshot)

        self._evaluate_subscriptions_and_notify()

    def _build_agent_snapshot_locked(self) -> AgentSnapshot:
        push = self.state.active_push
        component_id = push.component_id if push is not None else "unassigned"
        activity_id = push.activity_id if push is not None else None
        observations: dict[str, Any] = {}
        if push is not None:
            try:
                observations[component_id] = self.algorithm.environment.observe(
                    component_id
                )
            except Exception as exc:
                observations[component_id] = {"error": str(exc)}
        return AgentSnapshot(
            agent_id=self.entity_id,
            time=self.state.steps,
            location=None,
            activity_id=activity_id,
            component_observations=observations,
            capabilities=self.spec.capabilities,
        )

    def _send_agent_snapshot(self, snapshot: AgentSnapshot) -> None:
        if self.transport.sender is None:
            raise RuntimeError("agent transport sender is not initialized")
        controller = self.transport.config.controller_for_agent(self.entity_id)
        self.transport.sender.send(
            endpoint=controller.endpoint,
            kind=MessageKind.AGENT_SNAPSHOT,
            recipient_id=controller.controller_id,
            payload=snapshot,
        )

    def _evaluate_subscriptions_and_notify(self) -> None:
        with self._condition:
            subscriptions = list(self._subscriptions)
            notified_subs = self._notified_subscriptions

        if not subscriptions:
            return

        for sub in subscriptions:
            token = sub.get("token")
            if not token or token in notified_subs:
                continue

            cond_type = sub.get("condition_type")
            subscriber_id = sub.get("subscriber_id")
            repeat_count = sub.get("repeat_count", 3)

            if not subscriber_id:
                continue

            if subscriber_id not in self.transport.config.admins and not sub.get("subscriber_endpoint"):
                continue

            triggered = False
            if cond_type == "agent_steps":
                target_steps = sub.get("steps")
                if target_steps is None:
                    target_steps = sub.get("threshold", 0.0)
                if self.state.steps >= target_steps:
                    triggered = True

            if triggered:
                with self._condition:
                    if not hasattr(self, "_repeat_countdowns"):
                        self._repeat_countdowns = {}
                    if token not in self._repeat_countdowns:
                        self._repeat_countdowns[token] = repeat_count
                        self._notified_subscriptions.add(token)

        if hasattr(self, "_repeat_countdowns"):
            for token, remaining in list(self._repeat_countdowns.items()):
                if remaining <= 0:
                    continue

                target_sub = None
                for sub in subscriptions:
                    if sub.get("token") == token:
                        target_sub = sub
                        break

                if target_sub is None:
                    continue

                subscriber_id = target_sub.get("subscriber_id")
                admin_spec = self.transport.config.admins.get(subscriber_id)
                if admin_spec is not None:
                    sub_endpoint = admin_spec.endpoint
                elif target_sub.get("subscriber_endpoint") is not None:
                    endpoint_dict = target_sub.get("subscriber_endpoint")
                    sub_endpoint = EntityEndpoint(
                        host=endpoint_dict.get("host", "127.0.0.1"),
                        port=int(endpoint_dict.get("port", 0))
                    )
                else:
                    continue

                try:
                    self.transport.sender.send(
                        endpoint=sub_endpoint,
                        kind=MessageKind.STATUS_UPDATE,
                        recipient_id=subscriber_id,
                        payload={
                            "status": "triggered",
                            "condition_type": target_sub.get("condition_type"),
                            "token": f"{token}-{self.entity_id}",
                            "steps": self.state.steps,
                            "evaluation_return": self.algorithm.evaluation_return,
                        }
                    )
                except Exception as e:
                    self._record_error(e)

                self._repeat_countdowns[token] -= 1
                if self._repeat_countdowns[token] <= 0:
                    self._stop_event.set()
                    with self._condition:
                        self._condition.notify_all()

    def reset(self, context: ExperimentRunContext | None = None) -> None:
        super().reset(context)
        self.algorithm.reset()

    def _reset_role_state(self, context: ExperimentRunContext | None) -> None:
        self.state.latest_push = None
        self.state.active_push = None
        self.state.pending_policy_requests.clear()
        self.state.last_snapshot = None
        self._primary_resource_used = 0.0
        if hasattr(self, "_repeat_countdowns"):
            delattr(self, "_repeat_countdowns")

    def _role_results(self) -> dict[str, object]:
        simulator_stats_dict = {}
        primary_resource_usage = 0.0

        env_stats = getattr(self.algorithm.environment, "stats", None)
        if env_stats is not None:
            resource_types = env_stats.get_resource_types()
            for i, r_type in enumerate(resource_types):
                r_stats = env_stats.get_resource_stats(r_type)
                if r_stats is not None:
                    simulator_stats_dict[f"resource_{r_type}"] = {
                        "resource_initial": r_stats.resource_initial,
                        "resource_used": r_stats.resource_used,
                        "resource_available": r_stats.resource_available,
                    }
                    if i == 0:
                        primary_resource_usage = r_stats.resource_used

            for o_type in env_stats.get_operation_types():
                o_stats = env_stats.get_operation_stats(o_type)
                if o_stats is not None:
                    simulator_stats_dict[f"operation_{o_type}"] = {
                        "initial": o_stats.initial,
                        "performed": o_stats.performed,
                    }

        if not primary_resource_usage:
            primary_resource_usage = getattr(self, "_primary_resource_used", 0.0)

        res = {
            "steps": self.state.steps,
            "pull_requests_sent": self.state.pull_requests_sent,
            "trajectory_reports_sent": self.state.trajectory_reports_sent,
            "policy_cache_size": len(self.state.policy_cache_by_key),
            "pending_policy_requests": len(self.state.pending_policy_requests),
            "primary_resource_usage": primary_resource_usage,
            "evaluation_return": self.algorithm.evaluation_return,
        }
        if simulator_stats_dict:
            res["simulator_stats"] = simulator_stats_dict
        return res

    def _close_role_resources(self) -> None:
        self.transport.close()


@dataclass
class UdpPolicyManagerEntity(ThreadedUdpEntityBase):
    spec: PolicyManagerSpec = field(default=None)
    algorithm: SharedPolicyManagerPPO = field(default=None)
    config: RuntimeConfig = field(default=None)
    sender: RuntimeSender = field(default=None)
    trajectory_quorum: int = 2
    state: PolicyManagerRuntimeState = field(
        default_factory=PolicyManagerRuntimeState
    )
    _update_queue: queue.Queue[PolicyKey] = field(default_factory=queue.Queue, init=False)
    _update_thread: threading.Thread | None = field(default=None, init=False)

    @property
    def entity_id(self) -> str:
        return self.spec.policy_manager_id

    @property
    def kind(self) -> EntityKind:
        return EntityKind.POLICY_MANAGER

    @property
    def endpoint(self) -> EntityEndpoint:
        return self.spec.endpoint

    def handle_request(self, message: RuntimeMessage) -> bool:
        if message.kind == MessageKind.POLICY_REQUEST:
            if not isinstance(message.payload, PolicyRequest):
                raise TypeError("policy-request UDP message carried the wrong payload")
            self._handle_policy_request_locked(message.payload)
            return True
        if message.kind == MessageKind.TRAJECTORY_REPORT:
            report = message.payload
            if isinstance(report, TrajectoryUpdate):
                update = report
            elif isinstance(report, TrajectoryReport):
                update = TrajectoryUpdate(report=report)
            else:
                raise TypeError("trajectory UDP message carried the wrong payload")
            self._handle_trajectory_update_locked(update)
            return True
        return False

    def algorithm_loop(self) -> None:
        self._ensure_update_worker()
        while not self._stop_event.is_set():
            ready_keys: list[PolicyKey] = []
            with self._condition:
                for key, pool in self.state.trajectory_pools.items():
                    if (
                        key not in self.state.updates_in_flight
                        and len(pool) >= self.trajectory_quorum
                    ):
                        ready_keys.append(key)
                        self.state.updates_in_flight.add(key)
            for key in ready_keys:
                self._update_queue.put(key)
            time.sleep(self.loop_interval)

    def _ensure_update_worker(self) -> None:
        if self._update_thread is not None and self._update_thread.is_alive():
            return
        self._update_thread = threading.Thread(
            target=self._update_worker_loop,
            name=f"{self.entity_id}-policy-update-worker",
            daemon=True,
        )
        self._update_thread.start()

    def _update_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                key = self._update_queue.get(timeout=self.loop_interval)
            except queue.Empty:
                continue
            try:
                self._run_policy_update(key)
            finally:
                self._update_queue.task_done()

    def ingest_pull_request(self, request: PullRequest) -> bool:
        return False

    def run_update_round(self) -> PolicyUpdateStats:
        stats = self.algorithm.run_update_round()
        with self._condition:
            self.state.last_update_stats = stats
            self.state.policy_updates_completed += 1
        return stats

    def _handle_policy_request_locked(self, request: PolicyRequest) -> None:
        self.state.policy_requests_seen += 1
        self.state.waiters_by_key.setdefault(request.key, set()).add(request.agent_id)
        policy = self.state.policies_by_key.get(request.key)
        if policy is None:
            policy = self._create_policy_locked(request.key)
        self._send_policy_update(request.agent_id, policy)

    def _handle_trajectory_update_locked(self, update: TrajectoryUpdate) -> None:
        report = update.report
        key = PolicyKey(
            activity_id=report.descriptor.activity_id,
            capabilities=report.descriptor.capabilities,
        )
        self.state.trajectory_updates_seen += 1
        self.state.trajectory_pools.setdefault(key, []).append(update)
        if key not in self.state.policies_by_key:
            self._create_policy_locked(key)

    def _create_policy_locked(self, key: PolicyKey) -> PolicyUpdateNotification:
        policy_id = f"{key.activity_id}:{capability_key(key.capabilities)}"
        blob = self._serialize_policy(policy_id)
        policy = PolicyUpdateNotification(
            key=key,
            policy_id=policy_id,
            version=0,
            policy_blob=blob,
            metadata={"source": "initial"},
        )
        self.state.policies_by_key[key] = policy
        self.state.policies_by_id[policy_id] = policy
        return policy

    def _run_policy_update(self, key: PolicyKey) -> None:
        try:
            with self._condition:
                selected = list(self.state.trajectory_pools.get(key, ()))
            self.algorithm.pools.clear()
            for update in selected:
                self.algorithm.ingest_report(update.report)
            stats = self.algorithm.run_update_round()
            with self._condition:
                current = self.state.policies_by_key.get(key)
                version = 1 if current is None else current.version + 1
                policy_id = (
                    f"{key.activity_id}:{capability_key(key.capabilities)}"
                    if current is None
                    else current.policy_id
                )
                policy = PolicyUpdateNotification(
                    key=key,
                    policy_id=policy_id,
                    version=version,
                    policy_blob=self._serialize_policy(policy_id),
                    metadata={"source": "trajectory_merge"},
                )
                self.state.policies_by_key[key] = policy
                self.state.policies_by_id[policy_id] = policy
                self.state.policy_updates_completed += 1
                self.state.last_update_stats = stats
                waiters = set(self.state.waiters_by_key.get(key, set()))
                self.state.trajectory_pools.pop(key, None)
                self.state.updates_in_flight.discard(key)
                self.stats.algorithm_iterations += 1
            for agent_id in waiters:
                self._send_policy_update(agent_id, policy)
        except Exception as exc:
            with self._condition:
                self.state.updates_in_flight.discard(key)
            self._record_error(exc)

    def _serialize_policy(self, policy_id: str) -> bytes:
        serializer = getattr(self.algorithm.policy_bank, "serialize", None)
        if callable(serializer):
            return serializer(policy_id)
        return policy_id.encode("utf-8")

    def _send_policy_update(
        self,
        agent_id: str,
        update: PolicyUpdateNotification,
    ) -> None:
        agent = self.config.agents.get(agent_id)
        if agent is None:
            return
        self.sender.send(
            endpoint=agent.endpoint,
            kind=MessageKind.POLICY_UPDATE,
            recipient_id=agent_id,
            payload=update,
        )

    def reset(self, context: ExperimentRunContext | None = None) -> None:
        super().reset(context)
        self.algorithm.reset()

    def _reset_role_state(self, context: ExperimentRunContext | None) -> None:
        self.state.trajectory_pools.clear()
        self.state.waiters_by_key.clear()
        self.state.updates_in_flight.clear()

    def _role_results(self) -> dict[str, object]:
        return {
            "policy_requests_seen": self.state.policy_requests_seen,
            "trajectory_updates_seen": self.state.trajectory_updates_seen,
            "policy_updates_completed": self.state.policy_updates_completed,
            "policies": len(self.state.policies_by_key),
            "pending_pools": {
                str(key): len(pool)
                for key, pool in self.state.trajectory_pools.items()
            },
        }

    def _close_role_resources(self) -> None:
        if self._update_thread is not None and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        self.sender.close()


def _log_reset(entity_id: str, context: ExperimentRunContext | None) -> None:
    if context is None:
        LOGGER.info("reset entity %s", entity_id)
        return
    LOGGER.info(
        "reset entity %s for experiment context %s metadata=%s",
        entity_id,
        context.label,
        context.metadata or {},
    )


@dataclass
class UdpAdminEntity(ThreadedUdpEntityBase):
    spec: Any = field(default=None)
    admin_id: str = field(default="test-admin-1")
    entities: EntityAccessor = field(default=None)
    state: dict[str, Any] = field(default_factory=dict)
    received_notifications: list[RuntimeMessage] = field(default_factory=list)
    received_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    _processed_tokens: set[str] = field(default_factory=set)
    _sender: UdpRuntimeSender = field(default=None, init=False)

    @property
    def entity_id(self) -> str:
        return self.admin_id

    @property
    def kind(self) -> EntityKind:
        return EntityKind.ADMIN

    @property
    def endpoint(self) -> EntityEndpoint:
        return self.spec.endpoint

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sender = UdpRuntimeSender(self.entity_id)

    def handle_request(self, message: RuntimeMessage) -> bool:
        if message.kind == MessageKind.STATUS_UPDATE:
            token = message.payload.get("token")
            if token in self._processed_tokens:
                return True
            self._processed_tokens.add(token)
            self.received_notifications.append(message)
            return True
        if message.kind == MessageKind.LIFECYCLE_RESPONSE:
            response = message.payload
            if isinstance(response, EntityLifecycleResponse) and response.action == "results":
                if response.entity_id not in self.received_results:
                    self.received_results[response.entity_id] = response.response_payload
                    return True
        return False

    def preempt_all(self) -> None:
        for endpoint_proxy in self.entities.entity_endpoints():
            if endpoint_proxy.entity_id == self.entity_id:
                continue
            cmd = EntityLifecycleCommand(
                action="abort",
                entity_id=endpoint_proxy.entity_id,
                reply_endpoint=self.endpoint,
            )
            self._sender.send(
                endpoint=endpoint_proxy.endpoint,
                kind=MessageKind.LIFECYCLE_COMMAND,
                recipient_id=endpoint_proxy.entity_id,
                payload=cmd,
            )

    def request_results(self, entity_id: str) -> None:
        endpoint_proxy = self.entities.entity_endpoint(entity_id)
        if endpoint_proxy is None:
            return
        cmd = EntityLifecycleCommand(
            action="results",
            entity_id=entity_id,
            reply_endpoint=self.endpoint,
        )
        self._sender.send(
            endpoint=endpoint_proxy.endpoint,
            kind=MessageKind.LIFECYCLE_COMMAND,
            recipient_id=entity_id,
            payload=cmd,
        )

    def _reset_role_state(self, context: ExperimentRunContext | None) -> None:
        self.received_notifications.clear()
        self.received_results.clear()
        self._processed_tokens.clear()

    def _role_results(self) -> dict[str, object]:
        return {
            "received_notifications_count": len(self.received_notifications),
            "received_results_count": len(self.received_results),
        }

    def _close_role_resources(self) -> None:
        self._sender.close()

    def algorithm_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait(timeout=self.loop_interval)


import typing
if typing.TYPE_CHECKING:
    from runtime.entities import LogicalController, Agent, PolicyManager
_istype_UdpLogicalControllerEntity: typing.Type[LogicalController] = UdpLogicalControllerEntity
_istype_UdpAgentEntity: typing.Type[Agent] = UdpAgentEntity
_istype_UdpPolicyManagerEntity: typing.Type[PolicyManager] = UdpPolicyManagerEntity
