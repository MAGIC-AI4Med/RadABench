"""The real-environment layer.

165 real cases, executable tools, and the metrics that score them. A failure
here can belong to the agent, the tool, or the interface between them, and the
modules are arranged to keep those separable.

    from RealEnv import CaseManifest, build_default_pool, RealAgent

    manifest = CaseManifest.load("RealEnv/manifest.json")
    pool = build_default_pool()
"""

from .agent import AgentRun, RealAgent, StepRecord
from .cases import (BRATS_COHORTS, EXPECTED_COUNTS, TASKS, CaseManifest,
                    ManifestError, RealCase)
from .metrics import (accuracy, dice, jaccard, radgraph_f1, rate_score,
                      reported_ceiling_gap, set_jaccard, summarize_task)
from .tool_pool import (TOOL_FAMILIES, ExecutableTool, ExecutableToolPool,
                        build_default_pool, calibrate_from_results)

__all__ = [
    "CaseManifest", "RealCase", "TASKS", "EXPECTED_COUNTS", "BRATS_COHORTS",
    "ManifestError", "ExecutableTool", "ExecutableToolPool",
    "build_default_pool", "calibrate_from_results", "TOOL_FAMILIES",
    "RealAgent", "AgentRun", "StepRecord",
    "dice", "jaccard", "set_jaccard", "accuracy", "radgraph_f1", "rate_score",
    "reported_ceiling_gap", "summarize_task",
]
