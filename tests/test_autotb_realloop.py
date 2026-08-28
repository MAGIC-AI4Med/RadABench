"""End-to-end test of the AutoTB real-environment loop.

Proves the closure that was previously missing: a refusal leads to a build,
the built tool is registered, the case is replanned with it, and the outcome
is attributed to construction or usage rather than lumped together.

M3Builder is stubbed; what is under test is the control flow around it.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EvalPlat"))

import numpy as np  # noqa: E402

from AutoTB.m3builder import BuildOutcome, ToolPool  # noqa: E402
from RealEnv import RealAgent, RealCase, build_default_pool  # noqa: E402
from RealEnv.autotb_scenarios import (REQUIRED_CATEGORY, TIERS,  # noqa: E402
                                      build_scenarios, run_scenario)

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {label:<52} got={got!r:<22} want={want!r}")
    PASS, FAIL = PASS + ok, FAIL + (not ok)


CASE = RealCase("RGB-007", "report_generation", "RadGenome-BrainMRI",
                "Head & Neck", "MRI", "3D", "Write a report.",
                image_path="/tmp/x.nii.gz",
                reference={"finding": "mass", "impression": "glioma"})

REFUSAL = ("<NoCall><Purpose>need a report generator</Purpose>"
           "<Category>Report Generator</Category>"
           "<Anatomy>Head & Neck</Anatomy><Modality>MRI</Modality>"
           "<Ability>SpecificToolMissing</Ability></NoCall>")


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, **kw):
        return self.replies.pop(0) if self.replies else "done"


def executor(tool, inputs):
    return {key: np.ones((2, 2), dtype=bool) for key in tool.output}


class StubRunner:
    """Stands in for M3Builder; `succeeds` decides construction outcome."""

    def __init__(self, succeeds=True):
        self.succeeds, self.builds = succeeds, 0

    def build(self, card, dimensionality="2D"):
        self.builds += 1
        outcome = BuildOutcome(requested=card.as_dict())
        outcome.build_seconds = 12.0
        if self.succeeds:
            outcome.status = "SUCCESS"
            outcome.artifacts = ["/tmp/train.log", "/tmp/checkpoint.pt"]
        else:
            outcome.status = "CONSTRUCTION_FAILURE"
            outcome.stage_reached = "Setup Data"
            outcome.message = "dataset selection failed"
        return outcome

    def validate(self, outcome, card):
        outcome.validation = "PASSED" if self.succeeds else "NOT_RUN"
        return self.succeeds


print("── Scenario construction ──")
pool = build_default_pool()
scenarios = build_scenarios([CASE], pool, per_tier=4)
check("3 tiers x 4 = 12 scenarios", len(scenarios), 12)
check("tiers cover all three configs",
      sorted({s.condition for s in scenarios}), ["config1", "config2", "config3"])

cfg1 = next(s for s in scenarios if s.condition == "config1")
check("config1 removes the category entirely",
      len(cfg1.pool.candidates("Report Generator")), 0)

cfg2 = next(s for s in scenarios if s.condition == "config2")
check("config2 removes only this anatomy/modality",
      len(cfg2.pool.candidates("Report Generator", "Head & Neck", "MRI", "3D")), 0)
check("config2 leaves other coverage intact",
      len(cfg2.pool.candidates("Report Generator", "Chest", "X-ray", "2D")), 3)

cfg3 = next(s for s in scenarios if s.condition == "config3")
rg3 = cfg3.pool.candidates("Report Generator", "Head & Neck", "MRI", "3D")
check("config3 keeps the tools visible", len(rg3) > 0, True)
check("config3 marks the function unsupported",
      "missing_function" in rg3[0].unsupported, True)


def make_factory(second_pass_replies):
    """First pass refuses; second pass (post-build) uses the scripted replies."""
    state = {"calls": 0}

    def factory(tool_pool):
        state["calls"] += 1
        replies = [REFUSAL] if state["calls"] == 1 else list(second_pass_replies)
        return RealAgent(ScriptedLLM(replies), tool_pool, executor)

    return factory


print("\n── Build succeeds and the tool is used correctly -> SUCCESS ──")
with tempfile.TemporaryDirectory() as tmp:
    registry = ToolPool(str(Path(tmp) / "pool.json"))
    runner = StubRunner(succeeds=True)
    result = run_scenario(
        cfg1,
        make_factory([
            "<EndCall><Purpose>report</Purpose><Tool>AutoTB-RG-000</Tool>"
            "<Input>['$Image$']</Input></EndCall>",
            "A mass is present.",
        ]),
        runner, registry)
    check("AutoTB triggered", result.triggered, True)
    check("request grounded", result.grounded, True)
    check("M3Builder invoked once", runner.builds, 1)
    check("tool registered", result.registered_tool, "AutoTB-RG-000")
    check("downstream call succeeded", result.downstream_status, "SUCCESS")
    check("attributed SUCCESS", result.failure_mode, "SUCCESS")
    check("build time recorded", result.build_seconds, 12.0)
    check("registry persisted", len(registry.tools), 1)

print("\n── Build fails -> CONSTRUCTION_FAILURE, nothing registered ──")
with tempfile.TemporaryDirectory() as tmp:
    registry = ToolPool(str(Path(tmp) / "pool.json"))
    result = run_scenario(cfg1, make_factory(["irrelevant"]),
                          StubRunner(succeeds=False), registry)
    check("attributed to construction", result.failure_mode, "CONSTRUCTION_FAILURE")
    check("no tool registered", len(registry.tools), 0)
    check("no downstream attempt", result.downstream_status, "NOT_RUN")

print("\n── Built tool ignored on replan -> USAGE_ERROR, not construction ──")
with tempfile.TemporaryDirectory() as tmp:
    registry = ToolPool(str(Path(tmp) / "pool.json"))
    result = run_scenario(
        cfg1,
        make_factory([
            "<EndCall><Purpose>wrong tool</Purpose><Tool>AD-INS22-v1</Tool>"
            "<Input>['$Image$']</Input></EndCall>",
            "answer",
        ]),
        StubRunner(succeeds=True), registry)
    check("build succeeded", result.build_status, "SUCCESS")
    check("attributed to usage", result.failure_mode, "USAGE_ERROR")

print("\n── Agent never refuses -> PLANNING_ERROR, no build attempted ──")
with tempfile.TemporaryDirectory() as tmp:
    registry = ToolPool(str(Path(tmp) / "pool.json"))
    runner = StubRunner(succeeds=True)

    def never_refuses(tool_pool):
        return RealAgent(ScriptedLLM([
            "<EndCall><Purpose>x</Purpose><Tool>AD-INS22-v1</Tool>"
            "<Input>['$Image$']</Input></EndCall>",
            "answer",
        ]), tool_pool, executor)

    result = run_scenario(cfg1, never_refuses, runner, registry)
    check("attributed to planning", result.failure_mode, "PLANNING_ERROR")
    check("M3Builder never invoked", runner.builds, 0)

print("\n── Ungrounded request -> UNGROUNDED_REQUEST, no build ──")
with tempfile.TemporaryDirectory() as tmp:
    registry = ToolPool(str(Path(tmp) / "pool.json"))
    runner = StubRunner(succeeds=True)

    def wrong_category(tool_pool):
        return RealAgent(ScriptedLLM([
            "<NoCall><Purpose>x</Purpose><Category>Organ Segmentor</Category>"
            "<Anatomy>Head & Neck</Anatomy><Modality>MRI</Modality>"
            "<Ability>SpecificToolMissing</Ability></NoCall>",
        ]), tool_pool, executor)

    result = run_scenario(cfg1, wrong_category, runner, registry)
    check("triggered but ungrounded", result.triggered, True)
    check("attributed to grounding", result.failure_mode, "UNGROUNDED_REQUEST")
    check("M3Builder never invoked", runner.builds, 0)

print(f"\n{'=' * 76}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
