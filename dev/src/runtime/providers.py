from __future__ import annotations

from typing import Any

from core.algorithms.agent_expedition_ppo import (
    ActivityDescriptor,
    PPOTransition,
)
from core.algorithms.logical_controller import (
    AgentState,
    Assignment,
    EnvironmentSnapshot,
    IndicativeComponent,
    PullRequest,
    RedistributionPlan,
    RedistributionWeights,
    estimate_plan_costs,
    predict_efficacy_from_provider,
)
from core.distribution import (
    AStarMaxFlowAssignmentOptimizer,
    GreedyAssignmentOptimizer,
    RiskTransformerPipeline,
    RLMetaInterventionPolicy,
    RiskTransformerConfig,
)


def create_risk_transformer(
    model_path: str | None = None,
    device: str = "cpu",
) -> RiskTransformerPipeline:
    return RiskTransformerPipeline(
        config=RiskTransformerConfig(model_path=model_path, device=device)
    )


from .policy import InMemoryPolicy, InMemoryPolicyBank


class GATTaskModelProvider:
    def __init__(
        self,
        domain: str = "wildfire",
        ic_features: list[str] | None = None,
        agent_features: list[str] | None = None,
        feature_to_id: dict[str, int] | None = None,
        ic_dynamics_models: dict[str, str] | None = None,
        agent_efficacy_models: dict[str, str] | None = None,
        config: Any = None,
        factory: Any = None,
    ) -> None:
        self.domain = domain
        self.ic_features = list(ic_features) if ic_features else []
        self.agent_features = list(agent_features) if agent_features else []
        self.feature_to_id = dict(feature_to_id) if feature_to_id else {}
        self.ic_dynamics_models = dict(ic_dynamics_models) if ic_dynamics_models else {}
        self.agent_efficacy_models = dict(agent_efficacy_models) if agent_efficacy_models else {}
        self._efficacy_cache = {}
        self._dynamics_cache = {}

    def get_efficacy_model(self, ictype: int, meta: dict[str, Any] | None = None) -> Any:
        import os
        from core.taskmodel.agent_efficacy import GATAgentEfficacyModel

        key = str(ictype)
        if key in self._efficacy_cache:
            return self._efficacy_cache[key]

        model = None
        path = self.agent_efficacy_models.get(key)
        if path and os.path.exists(path):
            try:
                model = GATAgentEfficacyModel.load_pretrained(path)
            except Exception as e:
                print(f"Error loading pretrained GATAgentEfficacyModel from {path}: {e}")

        if model is None:
            active_features = self.ic_features[:3] if len(self.ic_features) >= 3 else self.ic_features
            model = GATAgentEfficacyModel(
                ic_type=ictype,
                active_feature_names=active_features,
                agent_feature_names=self.agent_features,
                feature_to_id=self.feature_to_id,
            )

        self._efficacy_cache[key] = model
        return model

    def get_icdynamics_model(self, ictype: int, meta: dict[str, Any] | None = None) -> Any:
        import os
        from core.taskmodel.ic_dynamics import GATICDynamicsModel

        key = str(ictype)
        if key in self._dynamics_cache:
            return self._dynamics_cache[key]

        model = None
        path = self.ic_dynamics_models.get(key)
        if path and os.path.exists(path):
            try:
                model = GATICDynamicsModel.load_pretrained(path)
            except Exception as e:
                print(f"Error loading pretrained GATICDynamicsModel from {path}: {e}")

        if model is None:
            model = GATICDynamicsModel(
                ic_type=ictype,
                feature_names=self.ic_features,
                feature_to_id=self.feature_to_id,
            )

        self._dynamics_cache[key] = model
        return model


import typing
if typing.TYPE_CHECKING:
    from runtime.factory import TaskModelProvider
    from core.algorithms.logical_controller import AssignmentOptimizer
_istype_GATTaskModelProvider: typing.Type[TaskModelProvider] = GATTaskModelProvider
_istype_GreedyAssignmentOptimizer: typing.Type[AssignmentOptimizer] = GreedyAssignmentOptimizer
_istype_AStarMaxFlowAssignmentOptimizer: typing.Type[AssignmentOptimizer] = AStarMaxFlowAssignmentOptimizer
