from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from core.algorithms.logical_controller import (
    AgentState,
    ComponentLocation,
    ComponentSize,
    EnvironmentSnapshot,
    IndicativeComponent,
)
from .config import RuntimeConfig
from .factory import DistributedEntitySystem, ExperimentRunContext
from .udp_factory import UdpSocketEntityFactory
from .testframework import (
    TestCase,
    TestInput,
    TestResult,
    SequenceTestProvider,
    TestSuite,
)


class FireRunner:
    pass


@dataclass(frozen=True)
class FireTestInput:
    fire_units: int
    indicative_components: int
    seed: int
    config: RuntimeConfig
    system: DistributedEntitySystem | None = None
    factory: UdpSocketEntityFactory | None = None
    description: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "description",
            f"Fire containment test with area={self.fire_units} (normalized), hotspots={self.indicative_components}, seed={self.seed}.",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "fire_units": self.fire_units,
            "indicative_components": self.indicative_components,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class IterationsTestInput:
    fire_units: int
    indicative_components: int
    seed: int
    iterations: int
    config: RuntimeConfig
    system: DistributedEntitySystem | None = None
    factory: UdpSocketEntityFactory | None = None
    description: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "description",
            f"Trajectory merging test with area={self.fire_units}, hotspots={self.indicative_components}, iterations={self.iterations}, seed={self.seed}.",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "fire_units": self.fire_units,
            "indicative_components": self.indicative_components,
            "iterations": self.iterations,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ScalabilityTestInput:
    agent_count: int
    seed: int
    config: RuntimeConfig
    system: DistributedEntitySystem | None = None
    factory: UdpSocketEntityFactory | None = None
    description: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "description",
            f"Scalability test with agents={self.agent_count}, seed={self.seed}.",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "agent_count": self.agent_count,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class AlgorithmTestInput:
    step_count: int
    seed: int
    config: RuntimeConfig
    system: DistributedEntitySystem | None = None
    factory: UdpSocketEntityFactory | None = None
    description: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "description",
            f"Algorithm performance test with steps={self.step_count}, seed={self.seed}.",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "step_count": self.step_count,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class FireTestResult:
    success: bool
    reason: str
    output_data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "output_data": self.output_data,
        }


class FireTestCase:
    def __init__(self, name: str, metric: str = "", runner: Any = None) -> None:
        self.name = name
        self.purpose = "Evaluate containment time and resources under wildfire constraints (Figures 3 and 4)."

    def run(self, test_input: TestInput) -> TestResult:
        assert isinstance(test_input, FireTestInput)
        import time

        total_time = 0.0
        total_resources = 0.0

        if test_input.system is not None and test_input.factory is not None:
            controller_ids = list(test_input.config.controllers.keys())
            controller_id = controller_ids[0] if controller_ids else "controller-1"

            admin_ids = list(test_input.config.admins.keys())
            admin_id = admin_ids[0] if admin_ids else "admin-1"

            admin_entity = test_input.factory.create_admin_entity(admin_id)
            admin_entity.start()

            try:
                token = f"{self.name}-{test_input.seed}-{time.time()}"
                implicit_subscription = {
                    "subscriber_id": admin_entity.entity_id,
                    "condition_type": "demand_reduction_percent",
                    "threshold": 95.0,
                    "token": token,
                    "repeat_count": 3,
                    "subscribe_entity": "logical_controller",
                }

                context = ExperimentRunContext(
                    label="wildfire_scenario",
                    metadata={
                        "fire_units": test_input.fire_units,
                        "indicative_components": test_input.indicative_components,
                        "demand": "normal",
                        "seed": test_input.seed,
                        "implicit_subscription": implicit_subscription,
                    },
                )

                test_input.system.reset(context)

                start_time = time.time()
                max_wait_seconds = 60.0
                termination_reason = "timeout"

                while time.time() - start_time < max_wait_seconds:
                    with admin_entity._condition:
                        if admin_entity.received_notifications:
                            termination_reason = "notification"
                            break
                        admin_entity._condition.wait(timeout=0.05)

                if termination_reason == "timeout":
                    admin_entity.preempt_all()

                target_ids = [controller_id] + list(test_input.config.agents.keys())
                for tid in target_ids:
                    admin_entity.request_results(tid)

                results_wait_timeout = 10.0
                results_start = time.time()
                while time.time() - results_start < results_wait_timeout:
                    with admin_entity._condition:
                        if all(tid in admin_entity.received_results for tid in target_ids):
                            break
                        admin_entity._condition.wait(timeout=0.05)

                controller_res = admin_entity.received_results.get(controller_id, {}).get("role", {})
                total_time = controller_res.get("containment_time", 0.0)

                for agent_id in test_input.config.agents.keys():
                    agent_res = admin_entity.received_results.get(agent_id, {}).get("role", {})
                    total_resources += agent_res.get("primary_resource_usage", agent_res.get("extinguishing_liquid_used", 0.0))

            finally:
                admin_entity.stop()

        output_data = {
            "total_time": total_time,
            "total_resources": total_resources,
            "fire_units": test_input.fire_units,
            "indicative_components": test_input.indicative_components,
            "seed": test_input.seed,
        }

        reason = f"Containment completed in {total_time:.2f}s using {total_resources:.2f} units of extinguishing liquid."
        return FireTestResult(
            success=True,
            reason=reason,
            output_data=output_data,
        )


class TrajectoryMergingTestCase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.purpose = "Evaluate trajectory merging performance and policy improvements"

    def run(self, test_input: TestInput) -> TestResult:
        assert isinstance(test_input, IterationsTestInput)
        import time

        total_time = 0.0
        total_resources = 0.0

        if test_input.system is not None and test_input.factory is not None:
            controller_ids = list(test_input.config.controllers.keys())
            controller_id = controller_ids[0] if controller_ids else "controller-1"

            admin_ids = list(test_input.config.admins.keys())
            admin_id = admin_ids[0] if admin_ids else "admin-1"

            admin_entity = test_input.factory.create_admin_entity(admin_id)
            admin_entity.start()

            try:
                token = f"{self.name}-{test_input.seed}-{time.time()}"
                implicit_subscription = {
                    "subscriber_id": admin_entity.entity_id,
                    "condition_type": "assignment_iterations",
                    "iterations": test_input.iterations,
                    "token": token,
                    "repeat_count": 3,
                    "subscribe_entity": "logical_controller",
                }

                context = ExperimentRunContext(
                    label="trajectory_merge_scenario",
                    metadata={
                        "fire_units": test_input.fire_units,
                        "indicative_components": test_input.indicative_components,
                        "demand": "normal",
                        "seed": test_input.seed,
                        "implicit_subscription": implicit_subscription,
                        "scale": 100,
                    },
                )

                test_input.system.reset(context)

                start_time = time.time()
                max_wait_seconds = 60.0
                termination_reason = "timeout"

                while time.time() - start_time < max_wait_seconds:
                    with admin_entity._condition:
                        if admin_entity.received_notifications:
                            termination_reason = "notification"
                            break
                        admin_entity._condition.wait(timeout=0.05)

                if termination_reason == "timeout":
                    admin_entity.preempt_all()

                target_ids = [controller_id] + list(test_input.config.agents.keys())
                for tid in target_ids:
                    admin_entity.request_results(tid)

                results_wait_timeout = 2.0
                results_start = time.time()
                while time.time() - results_start < results_wait_timeout:
                    with admin_entity._condition:
                        if all(tid in admin_entity.received_results for tid in target_ids):
                            break
                        admin_entity._condition.wait(timeout=0.05)

                controller_res = admin_entity.received_results.get(controller_id, {}).get("role", {})
                total_time = controller_res.get("containment_time", 0.0)

                for agent_id in test_input.config.agents.keys():
                    agent_res = admin_entity.received_results.get(agent_id, {}).get("role", {})
                    total_resources += agent_res.get("primary_resource_usage", agent_res.get("extinguishing_liquid_used", 0.0))

            finally:
                admin_entity.stop()

        output_data = {
            "total_time": total_time,
            "total_resources": total_resources,
            "fire_units": test_input.fire_units,
            "indicative_components": test_input.indicative_components,
            "seed": test_input.seed,
        }

        reason = f"Trajectory merging completed in {total_time:.2f}s using {total_resources:.2f} units of extinguishing liquid."
        return FireTestResult(
            success=True,
            reason=reason,
            output_data=output_data,
        )


class ScalabilityTestRunner:
    def __init__(self, name: str) -> None:
        self.name = name
        self.purpose = "Evaluate scalability and IC containment performance under varying agent counts."

    def run(self, test_input: TestInput) -> TestResult:
        assert isinstance(test_input, ScalabilityTestInput)
        import time

        contained_ics = 0.0
        percent_contained = 0.0
        initial_ic_count = 0

        if test_input.system is not None and test_input.factory is not None:
            controller_ids = list(test_input.config.controllers.keys())
            controller_id = controller_ids[0] if controller_ids else "controller-1"

            admin_ids = list(test_input.config.admins.keys())
            admin_id = admin_ids[0] if admin_ids else "admin-1"

            admin_entity = test_input.factory.create_admin_entity(admin_id)
            admin_entity.start()

            try:
                token = f"{self.name}-{test_input.seed}-{time.time()}"
                implicit_subscription = {
                    "subscriber_id": admin_entity.entity_id,
                    "condition_type": "agent_steps",
                    "steps": 2000,
                    "token": token,
                    "repeat_count": 3,
                    "subscribe_entity": "agent",
                }

                context = ExperimentRunContext(
                    label="scalability_scenario",
                    metadata={
                        "indicative_components": 4000,
                        "demand": "normal",
                        "seed": test_input.seed,
                        "implicit_subscription": implicit_subscription,
                    },
                )

                test_input.system.reset(context)

                start_time = time.time()
                max_wait_seconds = 120.0
                termination_reason = "timeout"

                while time.time() - start_time < max_wait_seconds:
                    with admin_entity._condition:
                        if admin_entity.received_notifications:
                            termination_reason = "notification"
                            break
                        admin_entity._condition.wait(timeout=0.05)

                if termination_reason == "timeout":
                    admin_entity.preempt_all()

                target_ids = [controller_id] + list(test_input.config.agents.keys())
                for tid in target_ids:
                    admin_entity.request_results(tid)

                results_wait_timeout = 2.0
                results_start = time.time()
                while time.time() - results_start < results_wait_timeout:
                    with admin_entity._condition:
                        if all(tid in admin_entity.received_results for tid in target_ids):
                            break
                        admin_entity._condition.wait(timeout=0.05)

                controller_res = admin_entity.received_results.get(controller_id, {}).get("role", {})
                initial_ic_count = controller_res.get("initial_ic_count") or 0
                percent_contained = controller_res.get("percent_contained") or 0.0
                contained_ics = initial_ic_count * (percent_contained / 100.0)

            finally:
                admin_entity.stop()

        output_data = {
            "initial_ic_count": initial_ic_count,
            "percent_contained": percent_contained,
            "contained_ics": contained_ics,
            "agent_count": test_input.agent_count,
            "seed": test_input.seed,
        }

        reason = f"Scalability run completed. Agents: {test_input.agent_count}, Contained ICs: {contained_ics:.2f} ({percent_contained:.2f}%)."
        return FireTestResult(
            success=True,
            reason=reason,
            output_data=output_data,
        )


class AlgorithmTestRunner:
    def __init__(self, name: str) -> None:
        self.name = name
        self.purpose = "Evaluate PPO algorithm performance (cumulative return) over a specific step count (Figure 6)."

    def run(self, test_input: TestInput) -> TestResult:
        assert isinstance(test_input, AlgorithmTestInput)
        import time

        total_return = 0.0

        if test_input.system is not None and test_input.factory is not None:
            controller_ids = list(test_input.config.controllers.keys())
            controller_id = controller_ids[0] if controller_ids else "controller-1"

            admin_ids = list(test_input.config.admins.keys())
            admin_id = admin_ids[0] if admin_ids else "admin-1"

            admin_entity = test_input.factory.create_admin_entity(admin_id)
            admin_entity.start()

            try:
                token = f"{self.name}-{test_input.seed}-{time.time()}"
                implicit_subscription = {
                    "subscriber_id": admin_entity.entity_id,
                    "condition_type": "agent_steps",
                    "steps": test_input.step_count,
                    "token": token,
                    "repeat_count": 3,
                    "subscribe_entity": "agent",
                }

                context = ExperimentRunContext(
                    label="ppo_algorithm_scenario",
                    metadata={
                        "indicative_components": 50,
                        "demand": "normal",
                        "seed": test_input.seed,
                        "implicit_subscription": implicit_subscription,
                    },
                )

                test_input.system.reset(context)

                start_time = time.time()
                max_wait_seconds = 60.0
                agent_count = len(test_input.config.agents)

                while time.time() - start_time < max_wait_seconds:
                    with admin_entity._condition:
                        if len(admin_entity.received_notifications) >= agent_count:
                            break
                        admin_entity._condition.wait(timeout=0.05)

                admin_entity.preempt_all()

                target_ids = [controller_id] + list(test_input.config.agents.keys())
                for tid in target_ids:
                    admin_entity.request_results(tid)

                results_wait_timeout = 2.0
                results_start = time.time()
                while time.time() - results_start < results_wait_timeout:
                    with admin_entity._condition:
                        if all(tid in admin_entity.received_results for tid in target_ids):
                            break
                        admin_entity._condition.wait(timeout=0.05)

                for agent_id in test_input.config.agents.keys():
                    agent_res = admin_entity.received_results.get(agent_id, {}).get("role", {})
                    total_return += agent_res.get("evaluation_return", 0.0)

            finally:
                admin_entity.stop()

        output_data = {
            "total_return": total_return,
            "step_count": test_input.step_count,
            "seed": test_input.seed,
        }

        reason = f"Algorithm PPO test completed. Steps: {test_input.step_count}, Total Cumulative Return: {total_return:.2f}."
        return FireTestResult(
            success=True,
            reason=reason,
            output_data=output_data,
        )


def run_containment_experiments(
    config: RuntimeConfig,
    system: DistributedEntitySystem,
    factory: UdpSocketEntityFactory,
    seeds: tuple[int, ...],
) -> None:
    fire_units_grid = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
    components_grid = (20, 30, 40, 50, 60, 70, 90, 110, 130, 150)

    inputs = [
        FireTestInput(
            fire_units=unit,
            indicative_components=comp,
            seed=seed,
            config=config,
            system=system,
            factory=factory,
        )
        for comp in components_grid
        for unit in fire_units_grid
        for seed in seeds
    ]

    runner = FireTestCase("FireContainmentTest")
    test_pairs = [(runner, test_input) for test_input in inputs]
    provider = SequenceTestProvider(test_pairs)

    suite = TestSuite("wildfire_containment_experiments", provider)
    suite.run("wildfire_experiments_results.json")


def run_trajectory_merging_experiments(
    config: RuntimeConfig,
    system: DistributedEntitySystem,
    factory: UdpSocketEntityFactory,
    seeds: tuple[int, ...],
) -> None:
    print("\n============================================================\nRUNNING TRAJECTORY MERGING SUITE\n============================================================")
    iterations_grid = (5, 10, 15, 20, 25, 30)
    iter_inputs = [
        IterationsTestInput(
            fire_units=500,
            indicative_components=30,
            seed=seed,
            iterations=it,
            config=config,
            system=system,
            factory=factory,
        )
        for it in iterations_grid
        for seed in seeds
    ]
    iter_runner = TrajectoryMergingTestCase("TrajectoryMergingTest")
    iter_pairs = [(iter_runner, test_input) for test_input in iter_inputs]
    iter_provider = SequenceTestProvider(iter_pairs)
    iter_suite = TestSuite("trajectory_merging_experiments", iter_provider)
    iter_suite.run("trajectory_merging_results.json")


def run_scalability_experiments(
    config: RuntimeConfig,
    system: DistributedEntitySystem,
    factory: UdpSocketEntityFactory,
    seeds: tuple[int, ...],
) -> None:
    print("\n============================================================\nRUNNING AGENT SCALABILITY SUITE\n============================================================")
    agent_counts = (10, 20, 30, 50, 100, 200, 400, 600, 800, 1000)
    all_agent_keys = list(config.agents.keys())

    if len(all_agent_keys) < 2:
        agent_counts = (len(all_agent_keys),) if all_agent_keys else (0,)

    for ac in agent_counts:
        if ac == 0:
            continue
        pruned_agents = {k: config.agents[k] for k in all_agent_keys[:ac]}
        temp_config = RuntimeConfig(
            controllers=config.controllers,
            agents=pruned_agents,
            policy_managers=config.policy_managers,
            admins=config.admins,
            physical_systems=config.physical_systems,
            components=config.components,
        )

        scal_inputs = [
            ScalabilityTestInput(
                seed=seed,
                agent_count=ac,
                config=temp_config,
                system=system,
                factory=factory,
            )
            for seed in seeds
        ]
        scal_runner = ScalabilityTestRunner("ScalabilityTest")
        scal_pairs = [(scal_runner, test_input) for test_input in scal_inputs]
        scal_provider = SequenceTestProvider(scal_pairs)
        scal_suite = TestSuite(f"scalability_experiments_agents_{ac}", scal_provider)
        scal_suite.run(f"scalability_results_agents_{ac}.json")


def run_algorithm_performance_experiments(
    config: RuntimeConfig,
    system: DistributedEntitySystem,
    factory: UdpSocketEntityFactory,
    seeds: tuple[int, ...],
) -> None:
    print("\n============================================================\nRUNNING ALGORITHM PERFORMANCE SUITE\n============================================================")
    steps_grid = (2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000)

    for steps in steps_grid:
        algo_inputs = [
            AlgorithmTestInput(
                step_count=steps,
                seed=seed,
                config=config,
                system=system,
                factory=factory,
            )
            for seed in seeds
        ]
        algo_runner = AlgorithmTestRunner("AlgorithmPPOTest")
        algo_pairs = [(algo_runner, test_input) for test_input in algo_inputs]
        algo_provider = SequenceTestProvider(algo_pairs)
        algo_suite = TestSuite(f"algorithm_ppo_experiments_steps_{steps}", algo_provider)
        algo_suite.run(f"algorithm_ppo_results_steps_{steps}.json")


def main() -> None:
    import argparse
    import json
    import logging
    import sys
    from pathlib import Path
    from .udp_factory import UdpSocketEntityFactory

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    parser = argparse.ArgumentParser(description="Run firefighting containment tests for Figures 3 and 4.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/runtime.experiment.json"),
        help="Runtime config JSON path.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds to run.",
    )
    args = parser.parse_args()

    try:
        config_payload = json.loads(args.config.read_text(encoding="utf-8"))
        config = RuntimeConfig.from_dict(config_payload)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    factory = UdpSocketEntityFactory(
        config=config,
        config_payload=config_payload,
    )

    print("Building and starting distributed entity system...")
    system = factory.create_distributed_system()
    system.start()

    try:
        seeds = tuple(range(args.seeds))
        run_containment_experiments(config, system, factory, seeds)
        run_trajectory_merging_experiments(config, system, factory, seeds)
        run_scalability_experiments(config, system, factory, seeds)
        run_algorithm_performance_experiments(config, system, factory, seeds)
    finally:
        system.stop()

if __name__ == "__main__":
    main()
