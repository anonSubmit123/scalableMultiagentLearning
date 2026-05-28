from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.capabilities import CapabilityProfile

from .logical_controller import (
    AgentId,
    ComponentId,
    ControllerPush,
    PullRequest,
)


PolicyId = str


class PolicyBankClient(Protocol):
    def get(self, descriptor: "ActivityDescriptor") -> "PolicyHandle":
        ...


class PolicyHandle(Protocol):
    policy_id: PolicyId
    version: int

    def sample_action(self, observation: Any, gain: float) -> tuple[Any, float]:
        ...

    def action_probability(self, observation: Any, action: Any, gain: float) -> float:
        ...

    def value(self, observation: Any, gain: float) -> float:
        ...

    def update(
        self,
        transition: "PPOTransition",
        advantage: float,
        target: float,
        probability_ratio: float,
        actor_loss: float,
        critic_loss: float,
    ) -> None:
        ...


class AgentEnvironment(Protocol):
    @property
    def stats(self) -> Any:
        ...

    def observe(self, component_id: ComponentId) -> Any:
        ...

    def execute(self, component_id: ComponentId, action: Any) -> "ExecutionResult":
        ...

    def reset(self) -> None:
        ...


class AgentTransport(Protocol):
    def send_pull_request(self, request: PullRequest) -> None:
        ...

    def send_trajectory_report(self, report: "TrajectoryReport") -> None:
        ...


@dataclass(frozen=True)
class ActivityDescriptor:
    activity_id: str
    capabilities: CapabilityProfile
    local_context: Any = None


@dataclass(frozen=True)
class ExecutionResult:
    reward: float
    next_observation: Any
    realized_efficacy: float
    terminal: bool = False


@dataclass(frozen=True)
class PPOTransition:
    observation: Any
    gain: float
    action: Any
    shaped_reward: float
    next_observation: Any
    mask: int
    old_action_probability: float


@dataclass(frozen=True)
class TrajectoryReport:
    agent_id: AgentId
    descriptor: ActivityDescriptor
    transitions: tuple[PPOTransition, ...]
    expected_efficacy: float
    marginal_gradient: float
    realized_efficacy: float
    policy_version: int


@dataclass(frozen=True)
class AgentStepResult:
    action: Any | None
    pull_requested: bool
    reported: bool
    shaped_reward: float | None = None


@dataclass
class AgentExpeditionGuidedPPO:
    agent_id: AgentId
    capabilities: CapabilityProfile
    policy_bank: PolicyBankClient
    environment: AgentEnvironment
    transport: AgentTransport
    discount: float = 0.99
    clip_epsilon: float = 0.20
    ppo_epochs: int = 4
    mismatch_reward_weight: float = 0.30
    gradient_reward_weight: float = 0.20
    efficacy_gap_threshold: float = 0.15
    report_interval: int = 25
    local_context: Any = None
    current_push: ControllerPush | None = None
    descriptor: ActivityDescriptor | None = None
    policy: PolicyHandle | None = None
    buffer: list[PPOTransition] = field(default_factory=list)
    last_realized_efficacy: float = 0.0
    step_count: int = 0
    evaluation_return: float = 0.0

    def receive_push(self, push: ControllerPush) -> None:
        self._report_and_clear_if_needed(force=True)
        self.current_push = push
        self.descriptor = ActivityDescriptor(
            activity_id=push.activity_id,
            capabilities=self.capabilities,
            local_context=self.local_context,
        )
        self.policy = self.policy_bank.get(self.descriptor)

    def step(self, force_report: bool = False) -> AgentStepResult:
        if self.current_push is None or self.descriptor is None or self.policy is None:
            return AgentStepResult(action=None, pull_requested=False, reported=False)

        self.step_count += 1
        push = self.current_push
        observation = self.environment.observe(push.component_id)
        action, old_probability = self.policy.sample_action(
            observation,
            push.marginal_gradient,
        )
        result = self.environment.execute(push.component_id, action)
        self.evaluation_return += result.reward
        mask = 0 if result.terminal else 1
        shaped_reward = shaped_ppo_reward(
            reward=result.reward,
            expected_efficacy=push.expected_efficacy,
            realized_efficacy=result.realized_efficacy,
            marginal_gradient=push.marginal_gradient,
            mismatch_reward_weight=self.mismatch_reward_weight,
            gradient_reward_weight=self.gradient_reward_weight,
        )

        transition = PPOTransition(
            observation=observation,
            gain=push.marginal_gradient,
            action=action,
            shaped_reward=shaped_reward,
            next_observation=result.next_observation,
            mask=mask,
            old_action_probability=old_probability,
        )
        self.buffer.append(transition)
        self._local_ppo_update(transition)

        efficacy_gap = max(push.expected_efficacy - result.realized_efficacy, 0.0)
        pull_requested = efficacy_gap > self.efficacy_gap_threshold
        if pull_requested:
            self.transport.send_pull_request(
                PullRequest(
                    agent_id=self.agent_id,
                    component_id=push.component_id,
                    efficacy_gap=efficacy_gap,
                    realized_efficacy=result.realized_efficacy,
                )
            )

        self.last_realized_efficacy = result.realized_efficacy
        reported = self._report_and_clear_if_needed(
            force=force_report or result.terminal,
        )
        return AgentStepResult(
            action=action,
            pull_requested=pull_requested,
            reported=reported,
            shaped_reward=shaped_reward,
        )

    def _local_ppo_update(self, transition: PPOTransition) -> None:
        if self.policy is None:
            return
        for _ in range(self.ppo_epochs):
            next_value = self.policy.value(
                transition.next_observation,
                transition.gain,
            )
            value = self.policy.value(transition.observation, transition.gain)
            advantage = (
                transition.shaped_reward
                + transition.mask * self.discount * next_value
                - value
            )
            target = transition.shaped_reward + (
                transition.mask * self.discount * next_value
            )
            probability = self.policy.action_probability(
                transition.observation,
                transition.action,
                transition.gain,
            )
            ratio = safe_ratio(probability, transition.old_action_probability)
            actor_loss = clipped_surrogate_loss(
                ratio,
                advantage,
                self.clip_epsilon,
            )
            value_loss = critic_loss(target, value)
            self.policy.update(
                transition,
                advantage,
                target,
                ratio,
                actor_loss,
                value_loss,
            )

    def _report_and_clear_if_needed(self, force: bool = False) -> bool:
        if not self.buffer or self.current_push is None or self.descriptor is None:
            return False
        should_report = force or (
            self.report_interval > 0 and self.step_count % self.report_interval == 0
        )
        if not should_report:
            return False

        policy_version = self.policy.version if self.policy is not None else 0
        self.transport.send_trajectory_report(
            TrajectoryReport(
                agent_id=self.agent_id,
                descriptor=self.descriptor,
                transitions=tuple(self.buffer),
                expected_efficacy=self.current_push.expected_efficacy,
                marginal_gradient=self.current_push.marginal_gradient,
                realized_efficacy=self.last_realized_efficacy,
                policy_version=policy_version,
            )
        )
        self.buffer.clear()
        return True

    def reset(self) -> None:
        self.current_push = None
        self.descriptor = None
        self.policy = None
        self.buffer.clear()
        self.last_realized_efficacy = 0.0
        self.step_count = 0
        self.evaluation_return = 0.0
        environment_reset = getattr(self.environment, "reset", None)
        if environment_reset is not None:
            environment_reset()


def shaped_ppo_reward(
    reward: float,
    expected_efficacy: float,
    realized_efficacy: float,
    marginal_gradient: float,
    mismatch_reward_weight: float,
    gradient_reward_weight: float,
) -> float:
    efficacy_gap = max(expected_efficacy - realized_efficacy, 0.0)
    return (
        reward
        - mismatch_reward_weight * efficacy_gap
        + gradient_reward_weight * marginal_gradient * realized_efficacy
    )


def clipped_surrogate_loss(
    probability_ratio: float,
    advantage: float,
    clip_epsilon: float,
) -> float:
    clipped_ratio = min(
        max(probability_ratio, 1.0 - clip_epsilon),
        1.0 + clip_epsilon,
    )
    return -min(probability_ratio * advantage, clipped_ratio * advantage)


def critic_loss(target: float, value: float) -> float:
    return (target - value) ** 2


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
