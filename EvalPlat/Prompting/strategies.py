"""The four prompting strategies, as composable modifiers on the agent loop.

Each targets a coordination bottleneck rather than the base model:

    backprop     a refined system prompt, from three rounds of automated
                 feedback plus manual adjustment over 220 QA pairs
    reflection   verify a tool's inputs before invoking it
    fewshot      exemplar workflows and tool templates
    multiagent   split the core into planner / executor / summarizer

They compose: --strategies fewshot,multiagent applies both, and the integrated
condition is all four.

A Strategy hooks three points: the system prompt, the message list before a
step, and the step output before it is parsed. Limiting them to these three
hooks is what keeps them independently ablatable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

__all__ = ["Strategy", "BackPropagation", "SelfReflection", "FewShot",
           "MultiAgent", "STRATEGIES", "compose", "parse_strategy_list"]

Message = Dict[str, str]


class Strategy:
    """Base class. Every hook is a no-op, so a strategy overrides only the
    stage it changes."""

    name = "base"
    description = ""

    def system_prompt(self, base: str) -> str:
        """Rewrite the system prompt before the run starts."""
        return base

    def before_step(self, messages: List[Message], state: Dict) -> List[Message]:
        """Adjust the message list before the next model call."""
        return messages

    def after_step(self, output: str, state: Dict,
                   chat: Callable[[List[Message]], str]) -> str:
        """Inspect or revise a step output before it is parsed.

        chat lets a strategy make its own model call; only self-reflection
        needs one.
        """
        return output


# --------------------------------------------------------------------------- #

BACKPROP_PREAMBLE = """\
Follow these rules, which were derived from analysing where radiology agents
most often fail:

1. Read every tool's declared anatomy, modality and dimensionality before
   selecting it. A tool whose scope does not cover this case is unusable even
   if its category is right.
2. Before each call, confirm every compulsory input already exists in the
   memory bank. Never pass a variable you have not produced.
3. Track what each step actually returned. Do not assume a step produced a
   value it did not.
4. When several tools fit the same category, prefer the one whose declared
   capability most specifically matches this case.
5. If no available tool can serve the next required step, stop and emit a
   NoCall stating exactly what is missing. Do not substitute a tool that
   almost fits, and do not invent a tool id.
6. The final step must produce the output type the question asked for. A
   correct chain that ends in the wrong output type has not answered the
   question.
"""


class BackPropagation(Strategy):
    """The refined prompt from three rounds of feedback (paper 2.2).

    The final prompt text is published because the procedure that produced it
    cannot be re-run without the same 220 QA pairs and manual passes.
    """

    name = "backprop"
    description = "Refined system prompt from iterative feedback"

    def system_prompt(self, base: str) -> str:
        return f"{base}\n\n{BACKPROP_PREAMBLE}"


# --------------------------------------------------------------------------- #

REFLECTION_PROMPT = """\
Before this call is executed, check it:

{call}

Memory bank currently holds: {available}

Answer on two lines.
VERDICT: OK or FIX
CORRECTED: if FIX, the corrected block in the same XML form; otherwise leave empty.

Check that every compulsory input exists in the memory bank, that no variable
is invented, and that the tool's declared anatomy and modality cover this case.
"""


class SelfReflection(Strategy):
    """Verify the call before executing it.

    Costs one extra model call per step. Catches invented variables and
    out-of-scope tools before they become runtime errors, but is unstable on
    its own, since it can also talk the agent out of a correct call.
    """

    name = "reflection"
    description = "Verify inputs before invoking a tool"

    def after_step(self, output: str, state: Dict, chat) -> str:
        if "<Call>" not in output and "<EndCall>" not in output:
            return output  # nothing to verify in a NoCall or a malformed reply

        verdict = chat([{"role": "user", "content": REFLECTION_PROMPT.format(
            call=output, available=sorted(state.get("memory", {})),
        )}])

        if "VERDICT: FIX" not in verdict.upper().replace("VERDICT:FIX", "VERDICT: FIX"):
            return output

        corrected = verdict.split("CORRECTED:", 1)[-1].strip()
        # Only accept a correction that is a call block. A verifier returning
        # prose must not be allowed to destroy a valid step.
        return corrected if "<" in corrected and ">" in corrected else output


# --------------------------------------------------------------------------- #

FEWSHOT_EXEMPLARS = """\
## Worked examples

### Example 1: organ mask annotation (chain: AC -> MC -> OS)
<Call>
  <Purpose>Identify the anatomical region</Purpose>
  <Tool>TOOL0</Tool>
  <Input>['$Image$']</Input>
</Call>
-> $Anatomy$ = Chest
<Call>
  <Purpose>Identify the imaging modality</Purpose>
  <Tool>TOOL1</Tool>
  <Input>['$Image$']</Input>
</Call>
-> $Modality$ = CT
<EndCall>
  <Purpose>Segment the requested organ</Purpose>
  <Tool>TOOL2</Tool>
  <Input>['$Image$', '$OrganObject$']</Input>
</EndCall>

### Example 2: the tool set is insufficient
The task needs a report generator; the pool has none that covers this
anatomy and modality.
<NoCall>
  <Purpose>Generate the radiology report</Purpose>
  <Category>Report Generator</Category>
  <Anatomy>Chest</Anatomy>
  <Modality>X-ray</Modality>
  <Ability>SpecificToolMissing</Ability>
</NoCall>

### Example 3: choosing between overlapping tools
Three report generators cover chest CT. Read each card's declared capability
and pick the one whose description matches this case most specifically,
rather than the first one listed.
"""


class FewShot(Strategy):
    """Exemplar workflows that standardise planning and I/O form.

    One of the two reliable wins (p < 0.05). The third exemplar matters as much
    as the other two: it demonstrates refusing, which agents otherwise rarely
    do.
    """

    name = "fewshot"
    description = "Exemplar workflows and tool templates"

    def system_prompt(self, base: str) -> str:
        return f"{base}\n\n{FEWSHOT_EXEMPLARS}"


# --------------------------------------------------------------------------- #

PLANNER_PROMPT = """\
You are the PLANNER. Given the query and the available tools, produce the
tool-category chain needed to answer it: categories only, in order, one per
line. Do not select concrete tools and do not execute anything.

Query: {query}

Available tool categories: {categories}
"""

EXECUTOR_PROMPT = """\
You are the EXECUTOR. The planner produced this chain:

{plan}

Execute it one step at a time, choosing the concrete tool for the current
category and managing inputs and outputs. Emit exactly one Call, EndCall, or
NoCall block per turn.
"""

SUMMARIZER_PROMPT = """\
You are the SUMMARIZER. The workflow produced these results:

{evidence}

Original question: {query}

Write the final answer. Use only the results above, and match the output type
the question asked for.
"""


@dataclass
class MultiAgent(Strategy):
    """Split the core into planner, executor and summarizer.

    The most consistent gain, and it helps where the single-core agent is
    weakest: long chains. Separating planning from execution stops a mid-chain
    error from rewriting the plan; separating summarization stops the
    executor's last tool output from being mistaken for the answer.

    Raises unsolvability awareness and grounding by roughly 5 points.
    """

    name = "multiagent"
    description = "Planner / executor / summarizer decomposition"

    def plan(self, chat, query: str, categories: Sequence[str]) -> str:
        return chat([{"role": "user", "content": PLANNER_PROMPT.format(
            query=query, categories=", ".join(categories))}])

    def executor_preamble(self, plan: str) -> Message:
        return {"role": "user", "content": EXECUTOR_PROMPT.format(plan=plan)}

    def summarize(self, chat, query: str, evidence: str) -> str:
        return chat([{"role": "user", "content": SUMMARIZER_PROMPT.format(
            query=query, evidence=evidence)}])

    def before_step(self, messages: List[Message], state: Dict) -> List[Message]:
        # Inject the plan once, on the first step.
        if state.get("plan") and not state.get("plan_injected"):
            state["plan_injected"] = True
            return messages + [self.executor_preamble(state["plan"])]
        return messages


# --------------------------------------------------------------------------- #

STRATEGIES = {
    BackPropagation.name: BackPropagation,
    SelfReflection.name: SelfReflection,
    FewShot.name: FewShot,
    MultiAgent.name: MultiAgent,
}


class _Composite(Strategy):
    """Applies several strategies in the order given."""

    name = "composite"

    def __init__(self, parts: Sequence[Strategy]) -> None:
        self.parts = list(parts)
        self.description = " + ".join(p.name for p in self.parts) or "none"

    def system_prompt(self, base: str) -> str:
        for part in self.parts:
            base = part.system_prompt(base)
        return base

    def before_step(self, messages, state):
        for part in self.parts:
            messages = part.before_step(messages, state)
        return messages

    def after_step(self, output, state, chat):
        for part in self.parts:
            output = part.after_step(output, state, chat)
        return output

    def multiagent(self) -> Optional[MultiAgent]:
        """The MultiAgent part, if present. The loop needs it by name."""
        return next((p for p in self.parts if isinstance(p, MultiAgent)), None)


def parse_strategy_list(spec: str) -> List[str]:
    """Turn "fewshot,multiagent" or "all" into validated names."""
    if not spec or spec.lower() in {"none", "off", ""}:
        return []
    if spec.lower() == "all":
        return list(STRATEGIES)

    names = [n.strip().lower() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise ValueError(
            f"Unknown strategy/strategies: {', '.join(unknown)}. "
            f"Available: {', '.join(STRATEGIES)}, or 'all'."
        )
    return names


def compose(spec: str) -> _Composite:
    """Build the composite strategy named by spec."""
    return _Composite([STRATEGIES[name]() for name in parse_strategy_list(spec)])
