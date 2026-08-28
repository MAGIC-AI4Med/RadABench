"""Real-environment agent loop: plan, execute against real tools, answer.

The executable counterpart of the synthetic pipelines. Here a tool call can
fail on its own terms (a poor mask, a missing checkpoint, a wrong input shape),
so the loop separates "the agent chose badly" from "the tool did badly" and
records both.
"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from RealEnv.tool_pool import ExecutableTool, ExecutableToolPool
from Utils.llm_client import LLMClient

__all__ = ["RealAgent", "AgentRun", "StepRecord"]

PLAN_PROMPT = """\
You are a radiological agent working with executable tools.

## Case
Anatomy: {anatomy}
Modality: {modality}
Dimensionality: {dimensionality}
Question: {question}

## Available tools
{tool_cards}

## Instructions
Select tools one at a time. Each tool's declared anatomy, modality and
dimensionality must match the case, and every compulsory input must already be
available in the memory bank.

Memory bank currently holds: {available}

Respond with exactly one block.

To call a tool:
<Call>
  <Purpose>what this step achieves</Purpose>
  <Tool>TOOL_ID</Tool>
  <Input>['$Variable$', ...]</Input>
</Call>

For the final step, use <EndCall> with the same fields.

If no available tool can serve the next required step, do not invent one:
<NoCall>
  <Purpose>what you needed to do</Purpose>
  <Category>the tool category that is missing</Category>
  <Anatomy>{anatomy}</Anatomy>
  <Modality>{modality}</Modality>
  <Ability>SpecificToolMissing or InsufficientCapability</Ability>
</NoCall>
"""

ANSWER_PROMPT = """\
## Case
{question}

## Tool outputs
{evidence}

Write the final answer to the clinical question, using only the tool outputs
above. Match the form the question asks for: a localisation description, a
diagnosis, or a radiology report.
"""

_CALL_RE = re.compile(r"<(Call|EndCall|NoCall)>(.*?)</\1>", re.DOTALL)
_FIELD_RE = "<{0}>(.*?)</{0}>"


@dataclass
class StepRecord:
    """One tool invocation, with enough detail to attribute a failure."""

    index: int
    purpose: str = ""
    tool_id: str = ""
    inputs: List[str] = field(default_factory=list)
    status: str = "OK"          # OK | TOOL_ERROR | INPUT_ERROR | UNKNOWN_TOOL
    outputs: Dict = field(default_factory=dict)
    message: str = ""


@dataclass
class AgentRun:
    """Everything one case produced."""

    case_id: str
    task: str
    steps: List[StepRecord] = field(default_factory=list)
    answer: str = ""
    refused: bool = False
    refusal_fields: Dict = field(default_factory=dict)
    error: str = ""

    @property
    def executed_chain(self) -> List[str]:
        return [s.tool_id for s in self.steps if s.status == "OK"]

    @property
    def steps_done(self) -> int:
        return sum(1 for s in self.steps if s.status == "OK")

    def as_dict(self) -> Dict:
        return {
            "case_id": self.case_id, "task": self.task, "answer": self.answer,
            "refused": self.refused, "refusal_fields": self.refusal_fields,
            "error": self.error,
            "steps": [s.__dict__ for s in self.steps],
        }


class RealAgent:
    """Drive one case end to end against an executable tool pool.

    executor is the function that runs a tool: it takes an ExecutableTool and
    the resolved inputs and returns a dict of outputs. Injecting it lets the
    loop be tested with stubs and run against M3Builder checkpoints, nnU-Net or
    RP3D-DM without changes here.
    """

    def __init__(self, llm: LLMClient, pool: ExecutableToolPool,
                 executor: Callable[[ExecutableTool, Dict], Dict],
                 max_steps: int = 10) -> None:
        self.llm = llm
        self.pool = pool
        self.executor = executor
        self.max_steps = max_steps

    def run(self, case) -> AgentRun:
        run = AgentRun(case_id=case.case_id, task=case.task)
        memory: Dict[str, object] = {
            "$Image$": case.image_path,
            "$Information$": case.question,
            # Anything the question itself states, e.g. the target organ.
            # Never the reference annotation.
            **dict(getattr(case, "inputs", {}) or {}),
        }

        # Show every card, not only the ones matching this case. Pre-filtering
        # would hand the agent the answer the Redundant and Insufficient
        # settings are meant to test.
        visible = [t.card() for t in self.pool]

        messages = [{"role": "user", "content": PLAN_PROMPT.format(
            anatomy=case.anatomy, modality=case.modality,
            dimensionality=case.dimensionality, question=case.question,
            tool_cards=json.dumps(visible, indent=2),
            available=sorted(memory),
        )}]

        for index in range(1, self.max_steps + 1):
            try:
                reply = self.llm.chat(messages)
            except Exception as error:  # noqa: BLE001
                run.error = f"LLM call failed at step {index}: {error}"
                return run

            messages.append({"role": "assistant", "content": reply})
            block = _CALL_RE.search(reply)
            if block is None:
                run.error = f"No <Call>/<EndCall>/<NoCall> block at step {index}."
                return run

            tag, content = block.group(1), block.group(2)

            if tag == "NoCall":
                run.refused = True
                run.refusal_fields = {
                    name: _field(content, name)
                    for name in ("Purpose", "Category", "Anatomy", "Modality", "Ability")
                }
                return run

            step = self._execute(index, content, memory)
            run.steps.append(step)

            if step.status != "OK":
                messages.append({"role": "user", "content":
                                 f"That step failed: {step.message}. Try a different tool."})
                continue

            memory.update(step.outputs)
            if tag == "EndCall":
                break
            messages.append({"role": "user", "content":
                             f"Result stored. Memory bank now holds: {sorted(memory)}"})

        run.answer = self._synthesize(case, memory)
        return run

    # -- one step -----------------------------------------------------------

    def _execute(self, index: int, content: str, memory: Dict) -> StepRecord:
        step = StepRecord(index=index, purpose=_field(content, "Purpose"))
        step.tool_id = _field(content, "Tool").strip()
        step.inputs = re.findall(r"\$[A-Za-z]+\$", _field(content, "Input"))

        tool = self.pool.by_id(step.tool_id)
        if tool is None:
            step.status = "UNKNOWN_TOOL"
            step.message = f"No tool with id {step.tool_id!r} in the pool."
            return step

        missing = [key for key in tool.compulsory_input if key not in memory]
        if missing:
            step.status = "INPUT_ERROR"
            step.message = f"Missing compulsory input(s): {', '.join(missing)}"
            return step

        resolved = {key: memory[key] for key in step.inputs if key in memory}
        try:
            step.outputs = self.executor(tool, resolved)
        except Exception as error:  # noqa: BLE001
            step.status = "TOOL_ERROR"
            step.message = f"{type(error).__name__}: {error}"
            step.outputs = {}
            traceback.clear_frames(error.__traceback__)
        return step

    def _synthesize(self, case, memory: Dict) -> str:
        evidence = {k: _brief(v) for k, v in memory.items() if k != "$Information$"}
        try:
            return self.llm.chat([{"role": "user", "content": ANSWER_PROMPT.format(
                question=case.question,
                evidence=json.dumps(evidence, indent=2, default=str),
            )}])
        except Exception as error:  # noqa: BLE001
            return f"[synthesis failed: {error}]"


def _field(content: str, name: str) -> str:
    match = re.search(_FIELD_RE.format(name), content, re.DOTALL)
    return match.group(1).strip() if match else ""


def _brief(value, limit: int = 400) -> str:
    """Keep array-shaped outputs out of the prompt; describe them instead."""
    if hasattr(value, "shape"):
        return f"<array shape={tuple(value.shape)}>"
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."
