from __future__ import annotations

import json
import time
import traceback
from typing import Any, Protocol, Sequence


class TestInput(Protocol):
    description: str

    def to_dict(self) -> dict[str, Any]:
        ...


class TestResult(Protocol):
    success: bool
    reason: str
    output_data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        ...


class TestCase(Protocol):
    name: str
    purpose: str

    def run(self, test_input: TestInput) -> TestResult:
        ...


class TestProvider(Protocol):
    def next(self) -> tuple[TestCase, TestInput] | None:
        ...


class SequenceTestProvider:
    def __init__(self, tests: Sequence[tuple[TestCase, TestInput]]) -> None:
        self._tests = list(tests)
        self._index = 0

    def next(self) -> tuple[TestCase, TestInput] | None:
        if self._index < len(self._tests):
            pair = self._tests[self._index]
            self._index += 1
            return pair
        return None


class TestSuite:
    def __init__(self, name: str, provider: TestProvider) -> None:
        self.name = name
        self.provider = provider

    def run(self, output_path: str | None = None) -> list[TestResult]:
        if output_path is None:
            output_path = f"{self.name}_results.json"

        results: list[TestResult] = []
        records: list[dict[str, Any]] = []

        start_time = time.time()
        attempted_count = 0
        successful_count = 0
        failed_count = 0

        print("=" * 60)
        print(f"RUNNING TEST SUITE: {self.name.upper()}")
        print("=" * 60)

        while True:
            pair = self.provider.next()
            if pair is None:
                break

            tc, ti = pair
            print(f"\nRunning TestCase: {tc.name}: {attempted_count + 1}")
            print(f"  - Purpose: {tc.purpose}")
            print(f"  - Input: {ti.description}")

            test_start = time.time()
            try:
                res = tc.run(ti)
                test_duration = time.time() - test_start
                results.append(res)

                attempted_count += 1
                if res.success:
                    successful_count += 1
                    status_str = "SUCCESS"
                else:
                    failed_count += 1
                    status_str = "FAILED"

                print(f"  - Result: {status_str} (took {test_duration:.4f}s)")
                print(f"  - Reason: {res.reason}")

                records.append({
                    "test_case": {
                        "name": tc.name,
                        "purpose": tc.purpose,
                    },
                    "test_input": ti.to_dict(),
                    "test_result": res.to_dict(),
                    "duration_seconds": test_duration,
                    "completed_successfully": True,
                    "error": None,
                    "traceback": None,
                })
            except Exception as e:
                test_duration = time.time() - test_start
                traceback_text = traceback.format_exc()
                failed_count += 1
                print(f"  - Result: ERROR (took {test_duration:.4f}s): {e}")
                print(traceback_text.rstrip())
                records.append({
                    "test_case": {
                        "name": tc.name,
                        "purpose": tc.purpose,
                    },
                    "test_input": ti.to_dict(),
                    "test_result": {
                        "success": False,
                        "reason": f"Execution error: {str(e)}",
                        "output_data": {},
                    },
                    "duration_seconds": test_duration,
                    "completed_successfully": False,
                    "error": str(e),
                    "traceback": traceback_text,
                })

        end_time = time.time()
        total_duration = end_time - start_time
        total_tests = successful_count + failed_count

        summary = {
            "suite_name": self.name,
            "total_tests": total_tests,
            "successful_tests": successful_count,
            "failed_tests": failed_count,
            "start_time": start_time,
            "end_time": end_time,
            "total_duration_seconds": total_duration,
            "tests": records,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        print("\n" + "=" * 60)
        print(f"SUITE COMPLETED: {successful_count}/{total_tests} passed (took {total_duration:.2f}s)")
        print(f"Results written to: {output_path}")
        print("=" * 60)

        return results
