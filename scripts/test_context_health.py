"""Offline experimental script to evaluate Context Health score behavior.

Tests the scoring engine across controlled synthetic scenarios to verify
monotonicity and behavioral consistency with the central hypothesis:

"For the same context utilization level, sessions with high irrelevant-context
density (low relevant-context ratio) should have worse Context Health than
sessions with low irrelevant-context density."

Important Disclaimer
--------------------
These experiments validate implementation behavior against predefined synthetic
scenarios. They do NOT validate the Context Health hypothesis on real coding-agent
sessions and do NOT establish predictive validity.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from context_health.health import ContextHealthResult, compute_context_health


@dataclass(frozen=True)
class Scenario:
    name: str
    utilization: float
    relevance: float
    complexity: float
    contradictions: int
    corrections: int
    retries: int


def run_scenario(scenario: Scenario) -> ContextHealthResult:
    """Execute a single synthetic scenario using the public health API."""
    return compute_context_health(
        context_utilization=scenario.utilization,
        relevant_context_ratio=scenario.relevance,
        task_complexity=scenario.complexity,
        contradiction_count=scenario.contradictions,
        correction_count=scenario.corrections,
        retry_count=scenario.retries,
    )


def print_table(results: list[tuple[Scenario, ContextHealthResult]]) -> None:
    """Print a clean formatted table of scenario results."""
    header = (
        f"{'Scenario':<10} | {'Utilization':<11} | {'Relevance':<9} | {'Complexity':<10} | "
        f"{'Contra':<6} | {'Corr':<5} | {'Retry':<5} | {'Score':<7} | {'Status':<10}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for sc, res in results:
        util_pct = f"{sc.utilization * 100:.0f}%"
        rel_pct = f"{sc.relevance * 100:.0f}%"
        comp_pct = f"{sc.complexity * 100:.0f}%"
        print(
            f"{sc.name:<10} | {util_pct:<11} | {rel_pct:<9} | {comp_pct:<10} | "
            f"{sc.contradictions:<6} | {sc.corrections:<5} | {sc.retries:<5} | "
            f"{res.health_score:6.2f} | {res.status:<10}"
        )

    print(separator)


def main() -> None:
    print("=" * 80)
    print("CONTEXT HEALTH V0.1 — SYNTHETIC BEHAVIORAL EXPERIMENTS")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Group A: Same utilization, varying relevance
    # -----------------------------------------------------------------------
    group_a = [
        Scenario("A1", 0.40, 0.95, 0.30, 0, 0, 0),
        Scenario("A2", 0.40, 0.80, 0.30, 0, 0, 0),
        Scenario("A3", 0.40, 0.60, 0.30, 0, 0, 0),
        Scenario("A4", 0.40, 0.40, 0.30, 0, 0, 0),
        Scenario("A5", 0.40, 0.20, 0.30, 0, 0, 0),
    ]

    # -----------------------------------------------------------------------
    # Group B: Same relevance, varying utilization
    # -----------------------------------------------------------------------
    group_b = [
        Scenario("B1", 0.20, 0.80, 0.30, 0, 0, 0),
        Scenario("B2", 0.40, 0.80, 0.30, 0, 0, 0),
        Scenario("B3", 0.60, 0.80, 0.30, 0, 0, 0),
        Scenario("B4", 0.80, 0.80, 0.30, 0, 0, 0),
        Scenario("B5", 0.95, 0.80, 0.30, 0, 0, 0),
    ]

    # -----------------------------------------------------------------------
    # Group C: Event degradation
    # -----------------------------------------------------------------------
    group_c = [
        Scenario("C1", 0.50, 0.80, 0.30, 0, 0, 0),
        Scenario("C2", 0.50, 0.80, 0.30, 1, 0, 0),
        Scenario("C3", 0.50, 0.80, 0.30, 1, 1, 1),
        Scenario("C4", 0.50, 0.80, 0.30, 3, 3, 3),
    ]

    # -----------------------------------------------------------------------
    # Group D: Combined degradation
    # -----------------------------------------------------------------------
    group_d = [
        Scenario("D1", 0.30, 0.90, 0.20, 0, 0, 0),
        Scenario("D2", 0.60, 0.60, 0.50, 1, 1, 1),
        Scenario("D3", 0.80, 0.40, 0.70, 2, 2, 2),
        Scenario("D4", 0.95, 0.20, 0.90, 4, 4, 4),
    ]

    all_scenarios = group_a + group_b + group_c + group_d
    results = [(sc, run_scenario(sc)) for sc in all_scenarios]
    res_dict = {sc.name: res for sc, res in results}

    print("\n[SCENARIO RESULTS]")
    print_table(results)

    # -----------------------------------------------------------------------
    # Automated Assertions
    # -----------------------------------------------------------------------
    print("\n[VERIFYING MONOTONIC ASSERTIONS]")

    try:
        # Group A Assertion: A1 > A2 > A3 > A4 > A5 (decreasing relevance)
        a_scores = [res_dict[f"A{i}"].health_score for i in range(1, 6)]
        for i in range(len(a_scores) - 1):
            assert a_scores[i] > a_scores[i + 1], (
                f"Group A monotonicity failed: A{i+1} ({a_scores[i]}) <= A{i+2} ({a_scores[i+1]})"
            )
        print("  ✓ Group A Passed: Strict monotonic decrease with decreasing relevance.")

        # Group B Assertion: B1 > B2 > B3 > B4 > B5 (increasing utilization)
        b_scores = [res_dict[f"B{i}"].health_score for i in range(1, 6)]
        for i in range(len(b_scores) - 1):
            assert b_scores[i] > b_scores[i + 1], (
                f"Group B monotonicity failed: B{i+1} ({b_scores[i]}) <= B{i+2} ({b_scores[i+1]})"
            )
        print("  ✓ Group B Passed: Strict monotonic decrease with increasing utilization.")

        # Group C Assertion: C1 > C2 > C3 > C4 (increasing negative events)
        c_scores = [res_dict[f"C{i}"].health_score for i in range(1, 5)]
        for i in range(len(c_scores) - 1):
            assert c_scores[i] > c_scores[i + 1], (
                f"Group C monotonicity failed: C{i+1} ({c_scores[i]}) <= C{i+2} ({c_scores[i+1]})"
            )
        print("  ✓ Group C Passed: Strict monotonic decrease with increasing error/correction events.")

        # Group D Assertion: D1 > D2 > D3 > D4 (combined degradation)
        d_scores = [res_dict[f"D{i}"].health_score for i in range(1, 5)]
        for i in range(len(d_scores) - 1):
            assert d_scores[i] > d_scores[i + 1], (
                f"Group D monotonicity failed: D{i+1} ({d_scores[i]}) <= D{i+2} ({d_scores[i+1]})"
            )
        print("  ✓ Group D Passed: Strict monotonic decrease across multi-dimensional degradation.")

    except AssertionError as err:
        print(f"\n❌ ASSERTION FAILED: {err}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Optional 2D Grid Experiment: Utilization vs Relevance
    # -----------------------------------------------------------------------
    print("\n[2D SYNTHETIC GRID: UTILIZATION x RELEVANCE]")
    util_steps = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rel_steps = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    grid_results: list[tuple[float, float, ContextHealthResult]] = []
    for u in util_steps:
        for r in rel_steps:
            res = compute_context_health(
                context_utilization=u,
                relevant_context_ratio=r,
                task_complexity=0.30,
                contradiction_count=0,
                correction_count=0,
                retry_count=0,
            )
            grid_results.append((u, r, res))

    best_config = max(grid_results, key=lambda x: x[2].health_score)
    worst_config = min(grid_results, key=lambda x: x[2].health_score)

    print(
        f"  Highest score: {best_config[2].health_score:.2f} [{best_config[2].status}] "
        f"at utilization={best_config[0]*100:.0f}%, relevance={best_config[1]*100:.0f}%"
    )
    print(
        f"  Lowest score:  {worst_config[2].health_score:.2f} [{worst_config[2].status}] "
        f"at utilization={worst_config[0]*100:.0f}%, relevance={worst_config[1]*100:.0f}%"
    )

    print("\nALL EXPERIMENTS PASSED")
    print("=" * 80)
    print(
        "These experiments validate implementation behavior against predefined synthetic\n"
        "scenarios. They do NOT validate the Context Health hypothesis on real coding-agent\n"
        "sessions and do NOT establish predictive validity."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
