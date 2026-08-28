"""Real insufficient-tool scenarios and the AutoTB loop over them.

660 scenarios are built from the real cases by hiding something the task needs,
in three tiers of increasing difficulty:

    Regular      (N=220)  config1   a required tool category is absent
    Intermediate (N=220)  config2   the category exists, not for this
                                    anatomy/modality
    Complex      (N=220)  config3   the capability itself is missing

The case is untouched, same image and same reference, so the task stays
clinically meaningful and only the tool environment becomes insufficient. The
agent has to notice, ask for the right thing, and then use what gets built.

This module supplies the replan_and_run closure AutoTB needs to finish
Algorithm 1. Without it the real layer stops at registration and can report
construction outcomes but never usage errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from AutoTB import AutoTBResult, run_real_autotb, summarize
from AutoTB.m3builder import M3BuilderRunner, ToolPool
from RealEnv.agent import RealAgent
from RealEnv.cases import RealCase
from RealEnv.tool_pool import ExecutableTool, ExecutableToolPool

__all__ = ["Scenario", "TIERS", "build_scenarios", "run_scenario",
           "run_all_scenarios"]

# Difficulty tier -> which insufficiency to induce.
TIERS = {
    "regular": "config1",
    "intermediate": "config2",
    "complex": "config3",
}

# The category each task requires. Hiding it makes the task unsolvable in a way
# the agent should be able to name.
REQUIRED_CATEGORY = {
    "organ_anomaly_grounding": "Anomaly Detector",
    "diagnosis_with_clues": "Disease Diagnoser",
    "report_generation": "Report Generator",
}


@dataclass
class Scenario:
    """One case paired with one induced insufficiency."""

    scenario_id: str
    case: RealCase
    tier: str
    condition: str
    hidden_category: str
    pool: ExecutableToolPool
    # What the verifier requires the agent to name.
    target_category: str = ""

    def as_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "case_id": self.case.case_id,
            "task": self.case.task,
            "tier": self.tier,
            "condition": self.condition,
            "hidden_category": self.hidden_category,
            "tools_remaining": len(self.pool),
        }


def build_scenarios(cases: Iterable[RealCase], full_pool: ExecutableToolPool,
                    per_tier: int = 220) -> List[Scenario]:
    """Instantiate the scenario set, cycling cases to reach the tier size.

    165 cases cannot fill 220 slots per tier, so cases repeat with different
    induced insufficiencies. The scenario id records which repetition it is, so
    results stay traceable to a case.
    """
    case_list = list(cases)
    if not case_list:
        raise ValueError("No cases supplied.")

    scenarios: List[Scenario] = []
    for tier, condition in TIERS.items():
        for index in range(per_tier):
            case = case_list[index % len(case_list)]
            category = REQUIRED_CATEGORY.get(case.task, "Report Generator")

            if condition == "config1":
                # Remove the category outright.
                pool = full_pool.without_category(category)
            elif condition == "config2":
                # Keep the category, remove this case's anatomy/modality.
                pool = full_pool.without_capability(category, case.anatomy, case.modality)
            else:
                # config3: the category is present and nominally in scope, but
                # every remaining variant declares the capability unsupported.
                pool = _strip_capability(full_pool, category, case)

            scenarios.append(Scenario(
                scenario_id=f"{tier}-{index:03d}-{case.case_id}",
                case=case, tier=tier, condition=condition,
                hidden_category=category, pool=pool,
                target_category=category,
            ))
    return scenarios


def _strip_capability(pool: ExecutableToolPool, category: str,
                      case: RealCase) -> ExecutableToolPool:
    """config3: leave the tools in place but mark the needed function absent.

    The hardest condition, because the category is still visible. An agent that
    only checks "is there a Report Generator?" answers yes and proceeds; only
    the capability constraints reveal the gap.
    """
    stripped: List[ExecutableTool] = []
    for tool in pool:
        if tool.category != category or not tool.supports(case.anatomy, case.modality):
            stripped.append(tool)
            continue
        limited = ExecutableTool(**tool.as_dict())
        limited.unsupported = {
            **tool.unsupported,
            "missing_function": [f"{case.task} for {case.anatomy} {case.modality}"],
        }
        limited.capability = (
            f"{tool.capability} Does NOT support "
            f"{case.task.replace('_', ' ')} on {case.anatomy} {case.modality}."
        )
        stripped.append(limited)
    return ExecutableToolPool(stripped)


# --------------------------------------------------------------------------- #
# running one scenario, end to end
# --------------------------------------------------------------------------- #

def run_scenario(scenario: Scenario, agent_factory: Callable[[ExecutableToolPool], RealAgent],
                 runner: M3BuilderRunner, pool_registry: ToolPool,
                 dimensionality: str = "") -> AutoTBResult:
    """Algorithm 1, closed: refuse, build, register, replan, run.

    agent_factory builds a RealAgent over a given tool pool. It is a factory
    rather than a fixed agent because the replan must see the augmented pool;
    reusing the original agent would replan against the same insufficient tools
    and the build would have achieved nothing.
    """
    # Pass 1: does the agent notice, and what does it ask for?
    first_run = agent_factory(scenario.pool).run(scenario.case)
    transcript = _as_transcript(first_run)

    def replan_and_run(tool_id: str) -> str:
        """Re-run the case with the newly built tool available."""
        built = pool_registry.tools[-1] if pool_registry.tools else {}
        augmented = ExecutableToolPool(list(scenario.pool) + [ExecutableTool(
            tool_id=tool_id,
            category=built.get("category", scenario.hidden_category),
            family=tool_id,
            dataset="AutoTB",
            anatomy=built.get("anatomy", scenario.case.anatomy),
            modality=built.get("modality", scenario.case.modality),
            dimensionality=dimensionality or scenario.case.dimensionality,
            capability=built.get("property", "AutoTB-built tool"),
            compulsory_input=built.get("compulsory_input", ["$Image$"]),
            output=built.get("output", []),
            source="AutoTB",
        )])
        second = agent_factory(augmented).run(scenario.case)

        if second.error:
            return "USAGE_ERROR"
        if second.refused:
            # Asked again despite having the tool: a grounding failure, not a
            # construction one.
            return "USAGE_ERROR"
        used = any(step.tool_id == tool_id for step in second.steps)
        if not used:
            return "USAGE_ERROR"
        failed = [s for s in second.steps if s.tool_id == tool_id and s.status != "OK"]
        return "USAGE_ERROR" if failed else "SUCCESS"

    return run_real_autotb(
        case_id=scenario.scenario_id,
        transcript=transcript,
        condition=scenario.condition,
        case={"Anatomy": scenario.case.anatomy, "Modality": scenario.case.modality},
        runner=runner,
        pool=pool_registry,
        target_category=scenario.target_category,
        replan_and_run=replan_and_run,
        dimensionality=dimensionality or scenario.case.dimensionality,
    )


def run_all_scenarios(scenarios: List[Scenario],
                      agent_factory: Callable[[ExecutableToolPool], RealAgent],
                      runner: M3BuilderRunner, pool_registry: ToolPool,
                      out_path: Optional[str] = None) -> Dict:
    """Run every scenario and report per-tier stability."""
    results: List[AutoTBResult] = []
    by_tier: Dict[str, List[AutoTBResult]] = {tier: [] for tier in TIERS}

    for scenario in scenarios:
        result = run_scenario(scenario, agent_factory, runner, pool_registry)
        results.append(result)
        by_tier[scenario.tier].append(result)
        print(f"  {scenario.scenario_id:<44} {result.failure_mode}")

    report = {
        "overall": summarize(results),
        "by_tier": {tier: summarize(rs) for tier, rs in by_tier.items()},
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps({
            "report": report,
            "scenarios": [r.__dict__ for r in results],
        }, indent=2, default=str))

    return report


def _as_transcript(run) -> str:
    """Render a run into the text the AutoTB verifier parses.

    The verifier reads <NoCall> blocks, so a refusal is rebuilt in that form
    rather than passed as a dict. One parser serves both layers.
    """
    if not run.refused:
        return "\n".join(
            f"<Call><Tool>{s.tool_id}</Tool></Call>" for s in run.steps
        ) or "<no output>"

    fields = run.refusal_fields
    return (
        "<NoCall>"
        f"<Purpose>{fields.get('Purpose', '')}</Purpose>"
        f"<Category>{fields.get('Category', '')}</Category>"
        f"<Anatomy>{fields.get('Anatomy', '')}</Anatomy>"
        f"<Modality>{fields.get('Modality', '')}</Modality>"
        f"<Ability>{fields.get('Ability', '')}</Ability>"
        "</NoCall>"
    )
