from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .agent_expedition_ppo import (
    ActivityDescriptor,
    PPOTransition,
    TrajectoryReport,
    clipped_surrogate_loss,
    critic_loss,
    safe_ratio,
)


PolicyId = str


class SharedPolicyBank(Protocol):
    def match_activity(
        self,
        activity_id: str,
        activity_type: str,
        capabilities: Mapping[str, Any] | frozenset[str] | list[str],
    ) -> str:
        ...

    def index(self, descriptor: ActivityDescriptor) -> tuple[PolicyId, int]:
        ...

    def action_probability(
        self,
        policy_id: PolicyId,
        observation: Any,
        action: Any,
        gain: float,
    ) -> float:
        ...

    def value(self, policy_id: PolicyId, observation: Any, gain: float) -> float:
        ...

    def update(
        self,
        policy_id: PolicyId,
        transition: PPOTransition,
        advantage: float,
        target: float,
        probability_ratio: float,
        actor_loss: float,
        critic_loss: float,
    ) -> None:
        ...


@dataclass(frozen=True)
class ScoredTrajectory:
    report: TrajectoryReport
    score: float
    return_sum: float


@dataclass(frozen=True)
class PolicyUpdateStats:
    pools_seen: int
    retained_trajectories: int
    replayed_trajectories: int
    transitions_seen: int


@dataclass
class StrategyFilter:
    success_rate: float = 0.20
    failure_rate: float = 0.10
    cap: int = 256

    def select(self, pool: list[ScoredTrajectory]) -> list[ScoredTrajectory]:
        if not pool:
            return []

        ordered = sorted(pool, key=lambda item: item.score)
        failure_count = int(len(pool) * self.failure_rate)
        success_count = int(len(pool) * self.success_rate)
        selected = ordered[:failure_count]
        if success_count:
            selected += ordered[-success_count:]

        deduped: list[ScoredTrajectory] = []
        seen: set[int] = set()
        for item in selected:
            marker = id(item)
            if marker not in seen:
                seen.add(marker)
                deduped.append(item)
        return deduped[: self.cap]


@dataclass
class SharedPolicyManagerPPO:
    policy_bank: SharedPolicyBank
    strategy_filter: StrategyFilter = field(default_factory=StrategyFilter)
    discount: float = 0.99
    clip_epsilon: float = 0.20
    ppo_epochs: int = 4
    critic_loss_weight: float = 0.5
    stale_version_threshold: int = 5
    beta_reward: float = 0.50
    beta_gradient: float = 0.30
    beta_mismatch: float = 0.20
    max_replay: int = 4
    replay_epsilon: float = 1e-9
    pools: dict[PolicyId, list[ScoredTrajectory]] = field(default_factory=dict)

    def ingest_report(self, report: TrajectoryReport) -> bool:
        policy_id, current_version = self.policy_bank.index(report.descriptor)
        if current_version - report.policy_version > self.stale_version_threshold:
            return False

        trajectory_return = sum(item.shaped_reward for item in report.transitions)
        score = trajectory_score(
            trajectory_return=trajectory_return,
            marginal_gradient=report.marginal_gradient,
            expected_efficacy=report.expected_efficacy,
            realized_efficacy=report.realized_efficacy,
            beta_reward=self.beta_reward,
            beta_gradient=self.beta_gradient,
            beta_mismatch=self.beta_mismatch,
        )
        self.pools.setdefault(policy_id, []).append(
            ScoredTrajectory(
                report=report,
                score=score,
                return_sum=trajectory_return,
            )
        )
        return True

    def run_update_round(self) -> PolicyUpdateStats:
        retained_trajectories = 0
        replayed_trajectories = 0
        transitions_seen = 0

        for policy_id, pool in list(self.pools.items()):
            retained = self.strategy_filter.select(pool)
            retained_trajectories += len(retained)
            mean_abs_score = _mean_abs_score(retained, self.replay_epsilon)

            replay_buffer: list[ScoredTrajectory] = []
            for item in retained:
                copies = 1 + min(
                    self.max_replay,
                    int(abs(item.score) / mean_abs_score),
                )
                replay_buffer.extend([item] * copies)
                replayed_trajectories += copies

            for item in replay_buffer:
                for transition in item.report.transitions:
                    transitions_seen += 1
                    self._update_policy(policy_id, transition)

        pools_seen = len([pool for pool in self.pools.values() if pool])
        self.pools.clear()
        return PolicyUpdateStats(
            pools_seen=pools_seen,
            retained_trajectories=retained_trajectories,
            replayed_trajectories=replayed_trajectories,
            transitions_seen=transitions_seen,
        )

    def _update_policy(
        self,
        policy_id: PolicyId,
        transition: PPOTransition,
    ) -> None:
        for _ in range(self.ppo_epochs):
            value = self.policy_bank.value(
                policy_id,
                transition.observation,
                transition.gain,
            )
            next_value = self.policy_bank.value(
                policy_id,
                transition.next_observation,
                transition.gain,
            )
            target = transition.shaped_reward + (
                transition.mask * self.discount * next_value
            )
            advantage = target - value
            probability = self.policy_bank.action_probability(
                policy_id,
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
            self.policy_bank.update(
                policy_id,
                transition,
                advantage,
                target,
                ratio,
                actor_loss,
                self.critic_loss_weight * value_loss,
            )

    def reset(self) -> None:
        self.pools.clear()
        bank_reset = getattr(self.policy_bank, "reset", None)
        if bank_reset is not None:
            bank_reset()


def trajectory_score(
    trajectory_return: float,
    marginal_gradient: float,
    expected_efficacy: float,
    realized_efficacy: float,
    beta_reward: float,
    beta_gradient: float,
    beta_mismatch: float,
) -> float:
    efficacy_gap = max(expected_efficacy - realized_efficacy, 0.0)
    return trajectory_return * (
        beta_reward
        + beta_gradient * marginal_gradient * realized_efficacy
        - beta_mismatch * efficacy_gap
    )


def _mean_abs_score(pool: list[ScoredTrajectory], epsilon: float) -> float:
    if not pool:
        return epsilon
    return sum(abs(item.score) for item in pool) / len(pool) + epsilon
