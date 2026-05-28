from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.algorithms.logical_controller import (
    Assignment,
    RedistributionPlan,
    estimate_plan_costs,
    predict_efficacy_from_provider,
    AssignmentContext,
)
from .astar_candidates import AStarCandidateGenerator
from .ic_map import ICMapBuilder
from .risk_prediction import TransformerRiskPredictor
from .risk_transformer import (
    RISK_TRANSFORMER_WINDOW,
    RiskTransformerConfig,
    RiskTransformerPipeline,
)
from .rl_meta_control import RLMetaInterventionPolicy


@dataclass
class GreedyAssignmentOptimizer:
    max_workload: float = 1.0
    config: Any = None

    def solve(self, context: AssignmentContext) -> RedistributionPlan:
        assignments: dict[str, Assignment] = {}
        global_efficacy = 0.0

        for agent_id, agent in sorted(context.snapshot.agents.items()):
            if agent.workload > self.max_workload:
                continue

            best_assignment: Assignment | None = None
            best_score = float("-inf")
            for component in context.snapshot.components.values():
                if not component.required_capabilities.issubset(agent.capabilities):
                    continue

                efficacy = predict_efficacy_from_provider(
                    context.task_model_provider,
                    agent,
                    component,
                    getattr(context, "config", None),
                )
                candidate = Assignment(
                    agent_id=agent_id,
                    activity_id=agent.activity_id,
                    component_id=component.component_id,
                )
                score = component.demand * efficacy
                if score > best_score:
                    best_score = score
                    best_assignment = candidate

            if best_assignment is not None:
                assignments[agent_id] = best_assignment
                global_efficacy += best_score

        costs = estimate_plan_costs(
            assignments,
            context.current_assignment,
            context.pull_requests,
        )
        return RedistributionPlan(
            assignments=assignments,
            global_efficacy=global_efficacy,
            costs=costs,
        )


@dataclass
class AStarMaxFlowAssignmentOptimizer:
    config: Any = None
    ic_map_builder: ICMapBuilder = field(default_factory=ICMapBuilder)
    risk_predictor: TransformerRiskPredictor = field(
        default_factory=TransformerRiskPredictor
    )
    risk_transformer: RiskTransformerPipeline = field(
        default_factory=RiskTransformerPipeline
    )
    intervention_policy: RLMetaInterventionPolicy = field(
        default_factory=RLMetaInterventionPolicy
    )
    candidate_generator: AStarCandidateGenerator = field(
        default_factory=AStarCandidateGenerator
    )
    _snapshot_window: deque = field(
        default_factory=lambda: deque(maxlen=RISK_TRANSFORMER_WINDOW),
        init=False,
    )

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        meta_policy_model_path: str | None = None,
        config: Any = None,
        ic_map_builder: ICMapBuilder | None = None,
        risk_predictor: TransformerRiskPredictor | None = None,
        risk_transformer: RiskTransformerPipeline | None = None,
        intervention_policy: RLMetaInterventionPolicy | None = None,
        candidate_generator: AStarCandidateGenerator | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.ic_map_builder = ic_map_builder or ICMapBuilder()
        self.risk_predictor = risk_predictor or TransformerRiskPredictor()
        self.candidate_generator = candidate_generator or AStarCandidateGenerator()

        if risk_transformer is not None:
            self.risk_transformer = risk_transformer
        else:
            self.risk_transformer = RiskTransformerPipeline(
                config=RiskTransformerConfig(
                    model_path=model_path,
                    device=device,
                )
            )

        if intervention_policy is not None:
            self.intervention_policy = intervention_policy
        else:
            self.intervention_policy = RLMetaInterventionPolicy(
                model_checkpoint_path=meta_policy_model_path
            )

        self._snapshot_window = deque(maxlen=RISK_TRANSFORMER_WINDOW)

    def solve(self, context: AssignmentContext) -> RedistributionPlan:
        self._snapshot_window.append(context.snapshot)
        fallback_ic_map = self.ic_map_builder.build(context.snapshot)
        progression = self.risk_transformer.predict(tuple(self._snapshot_window))
        ic_map = (
            progression.to_spatial_temporal_map()
            if progression.components
            else fallback_ic_map
        )
        prediction = self.risk_predictor.predict(ic_map)
        decision = self.intervention_policy.select(
            ic_map=ic_map,
            prediction=prediction,
            agent_count=len(context.snapshot.agents),
        )
        candidates = self.candidate_generator.generate(
            context=context,
            prediction=prediction,
            decision=decision,
        )

        plans: list[RedistributionPlan] = []
        target_components = set(
            decision.selected_components
            or prediction.prioritized_components[: decision.max_components]
        )

        for candidate in candidates:
            global_efficacy = 0.0
            for agent_id, assignment in candidate.assignments.items():
                if assignment.component_id in target_components:
                    agent = context.snapshot.agents.get(agent_id)
                    component = context.snapshot.components.get(assignment.component_id)
                    if agent and component:
                        eff = predict_efficacy_from_provider(
                            context.task_model_provider,
                            agent,
                            component,
                            getattr(context, "config", None),
                        )
                        global_efficacy += eff

            costs = estimate_plan_costs(
                candidate.assignments,
                context.current_assignment,
                context.pull_requests,
            )

            plans.append(
                RedistributionPlan(
                    assignments=candidate.assignments,
                    global_efficacy=global_efficacy,
                    costs=costs,
                )
            )

        if not plans:
            return RedistributionPlan(assignments={}, global_efficacy=0.0)

        return max(plans, key=lambda plan: plan.objective_value(context.weights))


import typing
if typing.TYPE_CHECKING:
    from core.algorithms.logical_controller import AssignmentOptimizer
_istype_GreedyAssignmentOptimizer: typing.Type[AssignmentOptimizer] = GreedyAssignmentOptimizer
_istype_AStarMaxFlowAssignmentOptimizer: typing.Type[AssignmentOptimizer] = AStarMaxFlowAssignmentOptimizer
