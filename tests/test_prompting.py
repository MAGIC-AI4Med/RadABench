"""Tests for the four prompting strategies and their composition."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EvalPlat"))

from Prompting import STRATEGIES, compose, parse_strategy_list  # noqa: E402
from Prompting.strategies import MultiAgent  # noqa: E402

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {label:<54} got={got!r:<20} want={want!r}")
    PASS, FAIL = PASS + ok, FAIL + (not ok)


BASE = "You are a radiological agent."

print("── Parsing ──")
check("four strategies registered", sorted(STRATEGIES),
      ["backprop", "fewshot", "multiagent", "reflection"])
check("'all' expands", sorted(parse_strategy_list("all")),
      ["backprop", "fewshot", "multiagent", "reflection"])
check("empty means none", parse_strategy_list(""), [])
check("'none' means none", parse_strategy_list("none"), [])
check("whitespace tolerated", parse_strategy_list(" fewshot , backprop "),
      ["fewshot", "backprop"])
try:
    parse_strategy_list("fewshot,nonsense")
    check("unknown strategy rejected", False, True)
except ValueError as error:
    check("unknown strategy rejected", "nonsense" in str(error), True)
    check("error lists the valid names", "fewshot" in str(error), True)

print("\n── Baseline is untouched ──")
check("no strategies leaves the prompt alone", compose("").system_prompt(BASE), BASE)

print("\n── Back-propagation ──")
bp = compose("backprop").system_prompt(BASE)
check("prompt grows", len(bp) > len(BASE), True)
check("keeps the original", BASE in bp, True)
check("states the refusal rule", "NoCall" in bp, True)
check("states the output-type rule", "output type" in bp, True)

print("\n── Few-shot ──")
fs = compose("fewshot").system_prompt(BASE)
check("includes a worked chain", "<EndCall>" in fs, True)
check("includes a refusal exemplar", "<NoCall>" in fs, True)
check("includes a tool-choice exemplar", "overlapping tools" in fs, True)

print("\n── Composition applies both ──")
both = compose("backprop,fewshot").system_prompt(BASE)
check("back-propagation rules present", "NoCall" in both, True)
check("few-shot exemplars present", "Worked examples" in both, True)
check("longer than either alone", len(both) > max(len(bp), len(fs)), True)

print("\n── Self-reflection ──")
calls = []


def chat_ok(messages):
    calls.append(messages)
    return "VERDICT: OK\nCORRECTED:"


def chat_fix(messages):
    calls.append(messages)
    return ("VERDICT: FIX\nCORRECTED: <Call><Purpose>p</Purpose>"
            "<Tool>TOOL2</Tool><Input>['$Image$']</Input></Call>")


def chat_prose(messages):
    calls.append(messages)
    return "VERDICT: FIX\nCORRECTED: I think you should reconsider this step."


refl = compose("reflection")
original = "<Call><Purpose>p</Purpose><Tool>TOOL9</Tool><Input>['$Ghost$']</Input></Call>"
state = {"memory": {"$Image$": "x"}}

check("verdict OK leaves the call unchanged",
      refl.after_step(original, state, chat_ok), original)
check("verdict FIX applies the correction",
      "TOOL2" in refl.after_step(original, state, chat_fix), True)
check("a prose 'correction' is rejected, original kept",
      refl.after_step(original, state, chat_prose), original)

calls.clear()
refl.after_step("<NoCall><Category>x</Category></NoCall>", state, chat_ok)
check("a NoCall is not sent for verification", len(calls), 0)

print("\n── Multi-agent ──")
ma = MultiAgent()
plan = ma.plan(lambda m: "AC\nMC\nOS", "segment the liver", ["AC", "MC", "OS"])
check("planner returns a category chain", plan, "AC\nMC\nOS")
check("summarizer is a separate call",
      ma.summarize(lambda m: "final", "q", "evidence"), "final")

state = {"plan": "AC\nMC\nOS"}
msgs = ma.before_step([{"role": "user", "content": "go"}], state)
check("plan injected on the first step", len(msgs), 2)
check("and only once", len(ma.before_step(msgs, state)), 2)

print("\n── The integrated condition ──")
integrated = compose("all")
check("all four composed", len(integrated.parts), 4)
check("multiagent reachable by name",
      isinstance(integrated.multiagent(), MultiAgent), True)
allp = integrated.system_prompt(BASE)
check("carries both prompt-level strategies",
      "Worked examples" in allp and "NoCall" in allp, True)

print(f"\n{'=' * 78}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
