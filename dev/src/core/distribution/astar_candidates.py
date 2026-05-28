from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Any

from core.algorithms.logical_controller import Assignment, estimate_plan_costs, predict_efficacy_from_provider, AssignmentContext
from .risk_prediction import RiskPrediction
from .rl_meta_control import InterventionDecision


@dataclass(frozen=True)
class DistributionCandidate:
    assignments: dict[str, Assignment]
    rationale: str


@dataclass
class Edge:
    to: int
    cap: int
    cost: float
    flow: int
    rev: int


class MinCostFlowSolver:
    def solve(
        self,
        agents: list[Any],
        components: list[str],
        capacities: dict[str, int],
        efficacy_matrix: dict[tuple[str, str], float],
        budget: int,
    ) -> float:
        emax = max(efficacy_matrix.values(), default=1.0)
        if emax <= 0.0:
            emax = 1.0

        n_agents = len(agents)
        n_comps = len(components)

        s = 0
        agent_offset = 1
        comp_offset = agent_offset + n_agents
        t_prime = comp_offset + n_comps
        t = t_prime + 1
        num_nodes = t + 1

        graph: list[list[Edge]] = [[] for _ in range(num_nodes)]

        def add_edge(u: int, v: int, cap: int, cost: float) -> None:
            forward = Edge(to=v, cap=cap, cost=cost, flow=0, rev=len(graph[v]))
            backward = Edge(to=u, cap=0, cost=-cost, flow=0, rev=len(graph[u]))
            graph[u].append(forward)
            graph[v].append(backward)

        for i in range(n_agents):
            add_edge(s, agent_offset + i, 1, 0.0)

        for i, agent in enumerate(agents):
            for j, comp_id in enumerate(components):
                efficacy = efficacy_matrix.get((agent.agent_id, comp_id), 0.0)
                add_edge(agent_offset + i, comp_offset + j, 1, emax - efficacy)

        for j, comp_id in enumerate(components):
            cap = capacities.get(comp_id, n_agents)
            add_edge(comp_offset + j, t_prime, cap, 0.0)

        add_edge(t_prime, t, budget, 0.0)

        total_ic_cap = sum(capacities.get(comp_id, n_agents) for comp_id in components)
        target_flow = min(budget, n_agents, total_ic_cap)
        if target_flow <= 0:
            return 0.0

        flow = 0
        total_cost = 0.0
        potentials = [0.0] * num_nodes

        while flow < target_flow:
            dist = [float("inf")] * num_nodes
            parent_node = [-1] * num_nodes
            parent_edge = [-1] * num_nodes

            dist[s] = 0.0
            pq = [(0.0, s)]

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for idx, edge in enumerate(graph[u]):
                    residual_cap = edge.cap - edge.flow
                    if residual_cap > 0:
                        v = edge.to
                        reduced_cost = edge.cost + potentials[u] - potentials[v]
                        if dist[u] + reduced_cost < dist[v] - 1e-9:
                            dist[v] = dist[u] + reduced_cost
                            parent_node[v] = u
                            parent_edge[v] = idx
                            heapq.heappush(pq, (dist[v], v))

            if dist[t] == float("inf"):
                break

            for i in range(num_nodes):
                if dist[i] < float("inf"):
                    potentials[i] += dist[i]

            curr = t
            while curr != s:
                p = parent_node[curr]
                idx = parent_edge[curr]
                edge = graph[p][idx]
                rev_idx = edge.rev
                rev_edge = graph[curr][rev_idx]

                edge.flow += 1
                rev_edge.flow -= 1
                curr = p

            flow += 1
            total_cost += potentials[t]

        best_gain = flow * emax - total_cost
        return max(0.0, best_gain)


@dataclass
class AStarState:
    assignments: dict[str, Assignment]
    g: float
    h: float
    f: float

    def __lt__(self, other: AStarState) -> bool:
        return self.f < other.f


@dataclass
class AStarCandidateGenerator:
    max_candidates: int = 8

    def generate(
        self,
        context: AssignmentContext,
        prediction: RiskPrediction,
        decision: InterventionDecision,
    ) -> list[DistributionCandidate]:
        target_components = set(
            decision.selected_components
            or prediction.prioritized_components[: decision.max_components]
        )
        if not target_components:
            return []

        baseline_commitment = 0
        for assign in context.current_assignment.values():
            if assign.component_id in target_components:
                baseline_commitment += 1

        B = max(1, int(decision.degree * baseline_commitment))

        def compute_deviation(state_assignments: dict[str, Assignment]) -> int:
            edits = 0
            all_agents = set(context.current_assignment.keys()) | set(state_assignments.keys())

            for agent_id in all_agents:
                current = context.current_assignment.get(agent_id)
                curr_ic = current.component_id if current else None

                state = state_assignments.get(agent_id)
                state_ic = state.component_id if state else None

                if curr_ic in target_components:
                    if state_ic in target_components:
                        if state_ic != curr_ic:
                            edits += 2
                    else:
                        edits += 1
                else:
                    if state_ic in target_components:
                        edits += 1
            return edits

        def compute_objective_value(state_assignments: dict[str, Assignment]) -> float:
            efficacy = 0.0
            for agent_id, assignment in state_assignments.items():
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
                        efficacy += eff

            costs = estimate_plan_costs(
                state_assignments,
                context.current_assignment,
                context.pull_requests,
            )

            return (
                efficacy
                - context.weights.movement * costs.movement
                - context.weights.switching * costs.switching
                - context.weights.unresolved_pull * costs.unresolved_pull
            )

        all_movable_agents = sorted(
            context.snapshot.agents.values(),
            key=lambda a: a.workload,
        )

        efficacy_matrix: dict[tuple[str, str], float] = {}
        for agent in all_movable_agents:
            for comp_id in target_components:
                component = context.snapshot.components[comp_id]
                if component.required_capabilities.issubset(agent.capabilities):
                    eff = predict_efficacy_from_provider(
                        context.task_model_provider,
                        agent,
                        component,
                        getattr(context, "config", None),
                    )
                    efficacy_matrix[(agent.agent_id, comp_id)] = eff
                else:
                    efficacy_matrix[(agent.agent_id, comp_id)] = 0.0

        def flow_heuristic(state_assignments: dict[str, Assignment], remaining_budget: int) -> float:
            if remaining_budget <= 0:
                return 0.0

            assigned_agents = set(
                agent_id for agent_id, assign in state_assignments.items()
                if assign.component_id in target_components
            )
            relaxed_movable = [
                agent for agent in all_movable_agents
                if agent.agent_id not in assigned_agents
            ]

            if not relaxed_movable:
                return 0.0

            capacities = {comp_id: len(relaxed_movable) for comp_id in target_components}

            solver = MinCostFlowSolver()
            best_gain = solver.solve(
                agents=relaxed_movable,
                components=list(target_components),
                capacities=capacities,
                efficacy_matrix=efficacy_matrix,
                budget=remaining_budget,
            )

            return -best_gain

        def expand_state(state: AStarState) -> list[AStarState]:
            successors = []

            for agent_id in list(state.assignments.keys()):
                next_assigns = dict(state.assignments)
                del next_assigns[agent_id]

                dev = compute_deviation(next_assigns)
                if dev <= B:
                    obj = compute_objective_value(next_assigns)
                    h = flow_heuristic(next_assigns, B - dev)
                    successors.append(
                        AStarState(
                            assignments=next_assigns,
                            g=-obj,
                            h=h,
                            f=-obj + h,
                        )
                    )

            reserved_agents = [
                agent for agent_id, agent in context.snapshot.agents.items()
                if agent_id not in state.assignments
            ]

            for agent in reserved_agents:
                for component in context.snapshot.components.values():
                    if not component.required_capabilities.issubset(agent.capabilities):
                        continue

                    next_assigns = dict(state.assignments)
                    next_assigns[agent.agent_id] = Assignment(
                        agent_id=agent.agent_id,
                        activity_id=agent.activity_id,
                        component_id=component.component_id,
                    )

                    dev = compute_deviation(next_assigns)
                    if dev <= B:
                        obj = compute_objective_value(next_assigns)
                        h = flow_heuristic(next_assigns, B - dev)
                        successors.append(
                            AStarState(
                                assignments=next_assigns,
                                g=-obj,
                                h=h,
                                f=-obj + h,
                            )
                        )
            return successors

        def _serialize_assignments(assignments: dict[str, Assignment]) -> str:
            parts = []
            for agent_id in sorted(assignments):
                assign = assignments[agent_id]
                parts.append(f"{agent_id}:{assign.component_id}:{assign.activity_id}")
            return "|".join(parts)

        s0_assignments = {
            agent_id: assign
            for agent_id, assign in context.current_assignment.items()
            if agent_id in context.snapshot.agents
        }

        s0_deviation = compute_deviation(s0_assignments)
        s0_obj = compute_objective_value(s0_assignments)
        s0_h = flow_heuristic(s0_assignments, B - s0_deviation)
        s0 = AStarState(
            assignments=s0_assignments,
            g=-s0_obj,
            h=s0_h,
            f=-s0_obj + s0_h,
        )

        open_queue = [s0]
        closed_set = set()

        best_states: list[AStarState] = [s0]

        expansions = 0
        max_expansions = 1000

        while open_queue and expansions < max_expansions:
            state = heapq.heappop(open_queue)

            state_signature = _serialize_assignments(state.assignments)
            if state_signature in closed_set:
                continue
            closed_set.add(state_signature)

            if not any(_serialize_assignments(bs.assignments) == state_signature for bs in best_states):
                best_states.append(state)
                best_states.sort(key=lambda s: s.g)
                best_states = best_states[:self.max_candidates * 2]

            dev = compute_deviation(state.assignments)
            if B - dev <= 0:
                continue

            successors = expand_state(state)
            for succ in successors:
                succ_signature = _serialize_assignments(succ.assignments)
                if succ_signature not in closed_set:
                    heapq.heappush(open_queue, succ)

            expansions += 1

        best_states.sort(key=lambda s: s.g)

        candidates: list[DistributionCandidate] = []
        for bs in best_states[:self.max_candidates]:
            obj_val = -bs.g
            candidates.append(
                DistributionCandidate(
                    assignments=bs.assignments,
                    rationale=f"A* search solution (objective={obj_val:.4f}, deviation={compute_deviation(bs.assignments)}/{B})",
                )
            )

        if not candidates:
            candidates.append(
                DistributionCandidate(
                    assignments=s0_assignments,
                    rationale="A* initial state fallback",
                )
            )

        return candidates
