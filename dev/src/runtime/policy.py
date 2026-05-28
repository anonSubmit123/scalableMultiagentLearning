from __future__ import annotations

import os
import pickle
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.algorithms.agent_expedition_ppo import ActivityDescriptor, PPOTransition
from core.capabilities import capability_key

LOGGER = logging.getLogger(__name__)


class InMemoryPolicy:
    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id
        self.version = 0

    def sample_action(self, observation: Any, gain: float) -> tuple[str, float]:
        return ("operate", 0.8)

    def action_probability(self, observation: Any, action: Any, gain: float) -> float:
        return 0.8 if action == "operate" else 0.2

    def value(self, observation: Any, gain: float) -> float:
        return min(1.0, 0.5 + 0.05 * gain)

    def update(
        self,
        transition: PPOTransition,
        advantage: float,
        target: float,
        probability_ratio: float,
        actor_loss: float,
        critic_loss: float,
    ) -> None:
        self.version += 1


class InMemoryPolicyBank:
    def __init__(self) -> None:
        self._policies: dict[str, InMemoryPolicy] = {}
        self._activity_to_policy_map: dict[str, str] = {}

    def match_activity(
        self,
        activity_id: str,
        activity_type: str,
        capabilities: Mapping[str, Any] | frozenset[str] | list[str],
    ) -> str:
        caps_str = capability_key(capabilities)
        policy_id = f"{activity_type}:{caps_str}"
        self._activity_to_policy_map[activity_id] = policy_id
        if policy_id not in self._policies:
            self._policies[policy_id] = InMemoryPolicy(policy_id)
        return policy_id

    def get(self, descriptor: ActivityDescriptor) -> InMemoryPolicy:
        policy_id, _ = self.index(descriptor)
        return self._policies[policy_id]

    def index(self, descriptor: ActivityDescriptor) -> tuple[str, int]:
        policy_id = self._activity_to_policy_map.get(descriptor.activity_id)
        if not policy_id:
            policy_id = (
                f"{descriptor.activity_id}:"
                f"{capability_key(descriptor.capabilities)}"
            )
        policy = self._policies.setdefault(policy_id, InMemoryPolicy(policy_id))
        return policy.policy_id, policy.version

    def action_probability(
        self,
        policy_id: str,
        observation: Any,
        action: Any,
        gain: float,
    ) -> float:
        return self._policies[policy_id].action_probability(observation, action, gain)

    def value(self, policy_id: str, observation: Any, gain: float) -> float:
        return self._policies[policy_id].value(observation, gain)

    def update(
        self,
        policy_id: str,
        transition: PPOTransition,
        advantage: float,
        target: float,
        probability_ratio: float,
        actor_loss: float,
        critic_loss: float,
    ) -> None:
        self._policies[policy_id].update(
            transition,
            advantage,
            target,
            probability_ratio,
            actor_loss,
            critic_loss,
        )

    def reset(self) -> None:
        self._policies.clear()


class PersistentPolicyBank:
    def __init__(
        self,
        policies: dict[str, dict[str, Any]] | None = None,
        save_interval: int = 5,
        mappings_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._policies: dict[str, InMemoryPolicy] = {}
        self._paths: dict[str, str] = {}
        self.save_interval = save_interval
        self._update_counts: dict[str, int] = {}
        self.mappings_path = mappings_path or "scratch/policy_mappings.pkl"
        self._activity_to_policy_map: dict[str, str] = {}

        if os.path.exists(self.mappings_path):
            try:
                with open(self.mappings_path, "rb") as f:
                    self._activity_to_policy_map = pickle.load(f)
                LOGGER.info(f"Loaded activity-to-policy mappings from {self.mappings_path}")
            except Exception as e:
                LOGGER.error(f"Failed to load activity-to-policy mappings from {self.mappings_path}: {e}")

        if policies:
            for policy_id, meta in policies.items():
                path = meta.get("path")
                if not path:
                    continue
                self._paths[policy_id] = path
                self._update_counts[policy_id] = 0

                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            policy = pickle.load(f)
                        if isinstance(policy, InMemoryPolicy):
                            self._policies[policy_id] = policy
                            LOGGER.info(f"Loaded persistent policy {policy_id} from {path} (version {policy.version})")
                        else:
                            LOGGER.warning(f"File at {path} did not contain InMemoryPolicy; creating new one.")
                    except Exception as e:
                        LOGGER.error(f"Failed to load persistent policy from {path}: {e}")

                if policy_id not in self._policies:
                    self._policies[policy_id] = InMemoryPolicy(policy_id)
                    self.save_policy(policy_id)

    def save_policy(self, policy_id: str) -> None:
        path = self._paths.get(policy_id)
        if not path:
            return
        policy = self._policies.get(policy_id)
        if not policy:
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(policy, f, protocol=pickle.HIGHEST_PROTOCOL)
            LOGGER.info(f"Successfully persisted policy {policy_id} to {path} (version {policy.version})")
        except Exception as e:
            LOGGER.error(f"Failed to save policy {policy_id} to {path}: {e}")

    def save_mappings(self) -> None:
        if not self.mappings_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.mappings_path)), exist_ok=True)
            with open(self.mappings_path, "wb") as f:
                pickle.dump(self._activity_to_policy_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            LOGGER.info(f"Successfully persisted activity-to-policy mappings to {self.mappings_path}")
        except Exception as e:
            LOGGER.error(f"Failed to save activity-to-policy mappings to {self.mappings_path}: {e}")

    def match_activity(
        self,
        activity_id: str,
        activity_type: str,
        capabilities: Mapping[str, Any] | frozenset[str] | list[str],
    ) -> str:
        caps_str = capability_key(capabilities)
        policy_id = f"{activity_type}:{caps_str}"
        self._activity_to_policy_map[activity_id] = policy_id

        if policy_id not in self._policies:
            self._policies[policy_id] = InMemoryPolicy(policy_id)
            self._update_counts[policy_id] = 0

        self.save_mappings()
        return policy_id

    def get(self, descriptor: ActivityDescriptor) -> InMemoryPolicy:
        policy_id, _ = self.index(descriptor)
        return self._policies[policy_id]

    def index(self, descriptor: ActivityDescriptor) -> tuple[str, int]:
        policy_id = self._activity_to_policy_map.get(descriptor.activity_id)
        if not policy_id:
            policy_id = (
                f"{descriptor.activity_id}:"
                f"{capability_key(descriptor.capabilities)}"
            )
        if policy_id not in self._policies:
            self._policies[policy_id] = InMemoryPolicy(policy_id)
            self._update_counts[policy_id] = 0
        policy = self._policies[policy_id]
        return policy.policy_id, policy.version

    def action_probability(
        self,
        policy_id: str,
        observation: Any,
        action: Any,
        gain: float,
    ) -> float:
        return self._policies[policy_id].action_probability(observation, action, gain)

    def value(self, policy_id: str, observation: Any, gain: float) -> float:
        return self._policies[policy_id].value(observation, gain)

    def update(
        self,
        policy_id: str,
        transition: PPOTransition,
        advantage: float,
        target: float,
        probability_ratio: float,
        actor_loss: float,
        critic_loss: float,
    ) -> None:
        if policy_id not in self._policies:
            self._policies[policy_id] = InMemoryPolicy(policy_id)
            self._update_counts[policy_id] = 0

        self._policies[policy_id].update(
            transition,
            advantage,
            target,
            probability_ratio,
            actor_loss,
            critic_loss,
        )

        if policy_id in self._paths:
            self._update_counts[policy_id] += 1
            if self._update_counts[policy_id] % self.save_interval == 0:
                self.save_policy(policy_id)

    def reset(self) -> None:
        self._policies.clear()
        self._update_counts.clear()
        for policy_id, path in self._paths.items():
            self._update_counts[policy_id] = 0
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        policy = pickle.load(f)
                    if isinstance(policy, InMemoryPolicy):
                        self._policies[policy_id] = policy
                except Exception:
                    pass
            if policy_id not in self._policies:
                self._policies[policy_id] = InMemoryPolicy(policy_id)


import typing
if typing.TYPE_CHECKING:
    from core.algorithms.shared_policy_manager_ppo import SharedPolicyBank
_istype_InMemoryPolicyBank: typing.Type[SharedPolicyBank] = InMemoryPolicyBank
_istype_PersistentPolicyBank: typing.Type[SharedPolicyBank] = PersistentPolicyBank
