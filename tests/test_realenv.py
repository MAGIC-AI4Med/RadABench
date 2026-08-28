"""Integration test for the real-environment layer.

Runs the whole loop (manifest, tool pool, agent, executor, metrics) against
stubs, so the wiring is exercised without a model call, a GPU, or credentialed
data. What it proves is that the pieces fit; what it cannot prove is that the
real tools are any good.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EvalPlat"))

import numpy as np  # noqa: E402

from RealEnv import (CaseManifest, ManifestError, RealAgent, RealCase,  # noqa: E402
                     accuracy, build_default_pool, dice, jaccard,
                     reported_ceiling_gap, set_jaccard, summarize_task)

PASS = FAIL = 0


def check(label, got, want, tol=1e-9):
    global PASS, FAIL
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    print(f"  {'✓' if ok else '✗'} {label:<54} got={got!r:<20} want={want!r}")
    PASS, FAIL = PASS + ok, FAIL + (not ok)


print("── Mask metrics ──")
a = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
b = np.array([[1, 0, 0], [0, 1, 1]], dtype=bool)
check("Dice, 2 of (3+3) overlap", dice(a, b), 2 * 2 / 6)
check("Jaccard, 2 of 4 union", jaccard(a, b), 0.5)
check("identical masks", dice(a, a), 1.0)
check("two empty masks score 1.0, not NaN",
      dice(np.zeros((2, 2), bool), np.zeros((2, 2), bool)), 1.0)

print("\n── Diagnosis metrics ──")
check("accuracy", accuracy(["glioma", "meningioma"], ["glioma", "glioma"]), 0.5)
check("set Jaccard", set_jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

print("\n── Idealized-executable gap ──")
check("delta = (U-C)/C x 100", reported_ceiling_gap(0.58, 0.503),
      (0.58 - 0.503) / 0.503 * 100)

print("\n── Tool pool ──")
pool = build_default_pool()
check("42 M3Builder variants + 2 external", len(pool), 44)
check("chest X-ray anomaly detectors",
      len(pool.candidates("Anomaly Detector", "Chest", "X-ray", "2D")), 3)
check("config1 removes a whole category",
      len(pool.without_category("Report Generator").candidates("Report Generator")), 0)
check("config2 keeps other coverage of the same category",
      len(pool.without_capability("Report Generator", "Chest", "X-ray")
          .candidates("Report Generator", "Head & Neck", "MRI", "3D")), 3)
check("tool card hides performance", "performance" in pool.tools[0].card(), False)
check("tool card shows it when asked", "performance" in pool.tools[0].card(True), True)

print("\n── Manifest validation refuses a short set ──")
try:
    CaseManifest([RealCase("x", "report_generation", "s", "Chest", "X-ray", "2D",
                           "q", reference={"finding": "f"})]).validate()
    check("short manifest rejected", False, True)
except ManifestError as error:
    check("short manifest rejected", "expected 55" in str(error), True)

print("\n── Agent loop against a stub executor ──")


class StubLLM:
    """Replays a fixed script, so the loop is tested and not the model."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, messages, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else "final answer text"


def stub_executor(tool, inputs):
    return {key: np.ones((2, 2), dtype=bool) for key in tool.output}


case = RealCase("RGB-001", "report_generation", "RadGenome-BrainMRI",
                "Head & Neck", "MRI", "3D", "Write a report.",
                image_path="/tmp/x.nii.gz",
                reference={"finding": "mass", "impression": "glioma"})

run = RealAgent(StubLLM([
    "<Call><Purpose>detect</Purpose><Tool>AD-INS22-v2</Tool>"
    "<Input>['$Image$']</Input></Call>",
    "<EndCall><Purpose>report</Purpose><Tool>RG-GenomBra-v3</Tool>"
    "<Input>['$Image$']</Input></EndCall>",
    "The scan shows a mass.",
]), pool, stub_executor).run(case)
check("two steps executed", run.steps_done, 2)
check("executed chain recorded", run.executed_chain,
      ["AD-INS22-v2", "RG-GenomBra-v3"])
check("final answer synthesized", run.answer, "The scan shows a mass.")
check("not refused", run.refused, False)

print("\n── The question's own variables reach the memory bank ──")
grounding = RealCase("CIG-001", "organ_anomaly_grounding", "Chest Imagenome",
                     "Chest", "X-ray", "2D",
                     "Locate the right lung and describe any abnormality.",
                     image_path="/tmp/x.png",
                     inputs={"$OrganObject$": "right lung"},
                     reference={"target_region": "right lung",
                                "lesion_annotation": "/tmp/mask.png"})
seg_llm = StubLLM([
    "<Call><Purpose>segment</Purpose><Tool>OS-nnUNet</Tool>"
    "<Input>['$Image$', '$OrganObject$']</Input></Call>",
    "<EndCall><Purpose>detect</Purpose><Tool>AD-XDet10-v3</Tool>"
    "<Input>['$Image$']</Input></EndCall>",
    "The right lung shows consolidation.",
])
run_g = RealAgent(seg_llm, pool, stub_executor).run(grounding)
check("segmentation runs with the question's target organ", run_g.steps[0].status, "OK")
check("both steps executed", run_g.steps_done, 2)

print("\n── A tool whose compulsory input is absent is INPUT_ERROR ──")
no_target = RealCase("CIG-002", "organ_anomaly_grounding", "Chest Imagenome",
                     "Chest", "X-ray", "2D", "Locate something.",
                     image_path="/tmp/x.png", reference={"target_region": "?"})
run_n = RealAgent(StubLLM([
    "<Call><Purpose>segment</Purpose><Tool>OS-nnUNet</Tool>"
    "<Input>['$Image$', '$OrganObject$']</Input></Call>",
    "<EndCall><Purpose>detect</Purpose><Tool>AD-XDet10-v3</Tool>"
    "<Input>['$Image$']</Input></EndCall>",
    "answer",
]), pool, stub_executor).run(no_target)
check("missing $OrganObject$ flagged", run_n.steps[0].status, "INPUT_ERROR")
check("names the missing variable", "$OrganObject$" in run_n.steps[0].message, True)

print("\n── Unknown tool id is caught, and the agent can recover ──")
run3 = RealAgent(StubLLM([
    "<Call><Purpose>x</Purpose><Tool>NOPE-v9</Tool><Input>['$Image$']</Input></Call>",
    "<EndCall><Purpose>report</Purpose><Tool>RG-GenomBra-v1</Tool>"
    "<Input>['$Image$']</Input></EndCall>",
    "answer",
]), pool, stub_executor).run(case)
check("unknown tool flagged", run3.steps[0].status, "UNKNOWN_TOOL")
check("agent recovers on the next step", run3.steps[1].status, "OK")
check("failed step excluded from executed chain", run3.executed_chain,
      ["RG-GenomBra-v1"])

print("\n── A tool that raises is TOOL_ERROR, attributable to the tool ──")


def broken_executor(tool, inputs):
    raise RuntimeError("checkpoint not found")


run4 = RealAgent(StubLLM([
    "<EndCall><Purpose>report</Purpose><Tool>RG-GenomBra-v1</Tool>"
    "<Input>['$Image$']</Input></EndCall>",
    "answer",
]), pool, broken_executor).run(case)
check("tool failure attributed to the tool", run4.steps[0].status, "TOOL_ERROR")
check("message preserved", "checkpoint not found" in run4.steps[0].message, True)

print("\n── NoCall short-circuits into a refusal AutoTB can consume ──")
run5 = RealAgent(StubLLM([
    "<NoCall><Purpose>need RG</Purpose><Category>Report Generator</Category>"
    "<Anatomy>Head & Neck</Anatomy><Modality>MRI</Modality>"
    "<Ability>SpecificToolMissing</Ability></NoCall>",
]), pool, stub_executor).run(case)
check("refused", run5.refused, True)
check("category captured for AutoTB", run5.refusal_fields["Category"],
      "Report Generator")
check("no steps executed", run5.steps_done, 0)

print("\n── Task summary ──")
summary = summarize_task("diagnosis_with_clues", [
    {"predicted": "glioma", "reference": "glioma", "llm_rating": 4.0},
    {"predicted": "meningioma", "reference": "glioma", "llm_rating": 2.0},
])
check("accuracy", summary["accuracy"], 0.5)
check("llm_rating averaged", summary["llm_rating"], 3.0)

print(f"\n{'=' * 78}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
