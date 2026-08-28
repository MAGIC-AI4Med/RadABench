"""AutoTB control flow (Algorithm 1) for both benchmark layers.

Synthetic layer:

    plan -> executable?  -> run
                         -> ground the gap -> verify -> simulate card -> replan

Real-environment layer:

    plan -> executable?  -> run
                         -> ground the gap -> specify card
                            -> M3Builder (up to K trials)
                            -> validate -> register -> replan -> run

Both are identical up to the point where a card exists. After that the
synthetic layer marks the card available and measures orchestration, while the
real layer trains a model and can fail in ways the synthetic layer cannot
express.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .tool_card import ToolCard, specify_tool_card
from .verifier import VerificationResult, verify

__all__ = ["AutoTBResult", "run_synthetic_autotb", "run_real_autotb",
           "FAILURE_MODES"]

# Attribution categories from the failure-mode analysis. They are kept apart
# because they call for different fixes: Planning Error is a prompting problem,
# Construction Failure a tooling problem, Usage Error an I/O-alignment problem.
FAILURE_MODES = {
    "PLANNING_ERROR": "The agent never emitted a build request.",
    "UNGROUNDED_REQUEST": "A request was made but named the wrong missing field.",
    "CONSTRUCTION_FAILURE": "M3Builder could not produce a validated tool.",
    "USAGE_ERROR": "The tool was built and registered, but called incorrectly.",
    "SUCCESS": "Built (or simulated), registered, and used correctly.",
}


@dataclass
class AutoTBResult:
    """One case's outcome, with enough detail to attribute the failure."""

    case_id: str
    layer: str                       # "synthetic" | "real"
    condition: str = ""
    triggered: bool = False          # AutoTB Trigger Rate
    grounded: bool = False           # request correctly grounded
    build_status: str = "NOT_RUN"
    registered_tool: str = ""
    downstream_status: str = "NOT_RUN"
    failure_mode: str = ""
    build_seconds: float = 0.0
    detail: Dict = field(default_factory=dict)

    @property
    def autotb_success(self) -> int:
        return int(self.failure_mode == "SUCCESS")


# --------------------------------------------------------------------------- #
# synthetic layer
# --------------------------------------------------------------------------- #

def run_synthetic_autotb(case_id: str, transcript: str, condition: str,
                         case: Dict, target_category: str = "") -> AutoTBResult:
    """Score one synthetic-layer case. Nothing is trained in this layer.

    The verified card is simulated as available so the remaining orchestration
    can be evaluated. Success here means the request was correctly grounded,
    not that a working tool exists.
    """
    check: VerificationResult = verify(transcript, condition, case, target_category)

    result = AutoTBResult(
        case_id=case_id, layer="synthetic", condition=condition,
        triggered=check.aware, grounded=check.grounded,
        detail={"reason": check.reason, "fields": check.request.fields},
    )

    if not check.aware:
        result.failure_mode = "PLANNING_ERROR"
        return result
    if not check.grounded:
        result.failure_mode = "UNGROUNDED_REQUEST"
        return result

    card = specify_tool_card(check.request)
    result.build_status = "SIMULATED"
    result.registered_tool = f"AutoTB-{card.category}"
    result.failure_mode = "SUCCESS"
    result.detail["tool_card"] = card.as_dict()
    result.detail["training_or_model_weights"] = "NOT_RUN"
    return result


# --------------------------------------------------------------------------- #
# real-environment layer
# --------------------------------------------------------------------------- #

def run_real_autotb(case_id: str, transcript: str, condition: str, case: Dict,
                    runner, pool, target_category: str = "",
                    replan_and_run: Optional[Callable[[str], str]] = None,
                    dimensionality: str = "2D") -> AutoTBResult:
    """Execute Algorithm 1 for one real case.

    runner: an M3BuilderRunner.
    pool: a ToolPool to register into.
    replan_and_run: called with the registered tool id; replans with the
    augmented pool and returns a status string. Pass None to stop after
    registration, when scoring construction alone.
    """
    check = verify(transcript, condition, case, target_category)
    result = AutoTBResult(
        case_id=case_id, layer="real", condition=condition,
        triggered=check.aware, grounded=check.grounded,
        detail={"reason": check.reason, "fields": check.request.fields},
    )

    if not check.aware:
        result.failure_mode = "PLANNING_ERROR"
        return result
    if not check.grounded:
        result.failure_mode = "UNGROUNDED_REQUEST"
        return result

    card = specify_tool_card(check.request, dimensionality)
    result.detail["tool_card"] = card.as_dict()

    # Reuse a tool already in the pool instead of rebuilding it.
    if pool.covers(card):
        result.build_status = "REUSED"
    else:
        outcome = runner.build(card, dimensionality)
        result.build_seconds = outcome.build_seconds
        result.detail["build"] = outcome.as_dict()
        if not runner.validate(outcome, card):
            result.build_status = outcome.status
            result.failure_mode = "CONSTRUCTION_FAILURE"
            return result
        checkpoint = next((a for a in outcome.artifacts if "checkpoint" in a.lower()), "")
        result.registered_tool = pool.register(card, checkpoint, outcome)
        result.build_status = "SUCCESS"

    if replan_and_run is None:
        result.failure_mode = "SUCCESS"
        return result

    status = replan_and_run(result.registered_tool)
    result.downstream_status = status
    result.failure_mode = "SUCCESS" if status == "SUCCESS" else "USAGE_ERROR"
    return result


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def summarize(results: List[AutoTBResult]) -> Dict[str, float]:
    """Aggregate into the reported AutoTB metrics."""
    n = len(results) or 1
    built = [r for r in results if r.build_seconds > 0]
    return {
        "n": len(results),
        "trigger_rate": sum(r.triggered for r in results) / n,
        "grounding_rate": sum(r.grounded for r in results) / n,
        "execution_success_rate": sum(r.autotb_success for r in results) / n,
        "avg_build_minutes": (
            sum(r.build_seconds for r in built) / len(built) / 60 if built else 0.0
        ),
        "failure_modes": {
            mode: sum(r.failure_mode == mode for r in results)
            for mode in FAILURE_MODES
        },
    }
