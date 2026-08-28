"""LLM Rating: the three-judge clinical-quality protocol.

Each real case is scored by three independent evaluators (GPT-5.2-Chat,
Gemini-3.0-Pro, DeepSeek-R1) on whether the answer is clinically plausible,
complete and aligned with the task target. The reported figure averages over
cases and evaluators:

    LLMRating = (1 / 3N) * sum_j sum_m r_{j,m}

Three judges rather than one matters because a single judge's preferences and
scoring bias would be indistinguishable from a real quality difference between
systems. Per-judge scores and inter-judge disagreement are kept in the output
so a split panel stays visible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence

from Utils.llm_client import LLMClient

__all__ = ["JUDGES", "RATING_SCALE", "CaseRating", "LLMRater", "build_prompt"]

# The evaluator panel, by alias in Configs/models.yaml.
JUDGES = ("gpt-5.2-chat", "gemini-3.0-pro", "deepseek-r1")

RATING_SCALE = {
    1: "Clinically wrong or unusable; would mislead a reader.",
    2: "Partially relevant but with significant omissions or errors.",
    3: "Broadly correct and usable, with minor gaps.",
    4: "Clinically sound, complete, and well aligned with the task target.",
    5: "Matches what an experienced radiologist would produce.",
}

TASK_TARGETS = {
    "organ_anomaly_grounding":
        "localise the requested organ and any abnormality, and describe both",
    "diagnosis_with_clues":
        "state the diagnosis, justified by the grounding evidence provided",
    "report_generation":
        "produce a radiology report covering the findings and impression",
}

_SYSTEM = (
    "You are an experienced radiologist reviewing the output of an AI system. "
    "Judge only clinical quality: is the answer plausible, complete, and "
    "aligned with what the task asked for? Ignore style, verbosity, and "
    "formatting. Be strict: a fluent answer that misses the finding is a low "
    "score, not a middling one."
)


def build_prompt(task: str, question: str, answer: str, reference: str) -> List[Dict[str, str]]:
    """Build the evaluator prompt.

    The reference is shown last and labelled so the judge grades against it
    rather than against its own prior about the case.
    """
    target = TASK_TARGETS.get(task, "answer the clinical question")
    scale = "\n".join(f"  {score}: {text}" for score, text in RATING_SCALE.items())
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"""\
## Task
{task.replace('_', ' ')}: the system was asked to {target}.

## Clinical question
{question}

## System answer
{answer}

## Reference
{reference}

## Rating scale
{scale}

Respond with exactly two lines:
SCORE: <integer 1-5>
REASON: <one sentence>"""},
    ]


@dataclass
class CaseRating:
    """One case's scores, kept per judge."""

    case_id: str
    task: str
    scores: Dict[str, Optional[int]] = field(default_factory=dict)
    reasons: Dict[str, str] = field(default_factory=dict)

    @property
    def valid_scores(self) -> List[int]:
        return [s for s in self.scores.values() if s is not None]

    @property
    def mean_score(self) -> Optional[float]:
        scores = self.valid_scores
        return mean(scores) if scores else None

    @property
    def disagreement(self) -> Optional[float]:
        """Spread across judges. A high value means the mean hides a split."""
        scores = self.valid_scores
        return pstdev(scores) if len(scores) > 1 else 0.0 if scores else None


class LLMRater:
    """Runs the panel over a set of cases."""

    def __init__(self, judges: Sequence[str] = JUDGES) -> None:
        self.judges = list(judges)
        self._clients: Dict[str, LLMClient] = {}

    def _client(self, alias: str) -> LLMClient:
        if alias not in self._clients:
            self._clients[alias] = LLMClient(alias)
        return self._clients[alias]

    def rate_case(self, case_id: str, task: str, question: str,
                  answer: str, reference: str) -> CaseRating:
        rating = CaseRating(case_id=case_id, task=task)
        messages = build_prompt(task, question, answer, reference)

        for alias in self.judges:
            try:
                reply = self._client(alias).chat(messages)
                rating.scores[alias] = _parse_score(reply)
                rating.reasons[alias] = _parse_reason(reply)
            except Exception as error:  # noqa: BLE001
                # An unreachable judge is recorded as absent, not as a zero.
                # A failed API call is not a bad answer.
                rating.scores[alias] = None
                rating.reasons[alias] = f"[evaluator unavailable: {error}]"

        return rating

    def rate_all(self, cases: List[Dict]) -> List[CaseRating]:
        """Score many cases. Each dict needs case_id, task, question, answer,
        reference."""
        return [
            self.rate_case(c["case_id"], c["task"], c["question"],
                           c["answer"], c["reference"])
            for c in cases
        ]

    @staticmethod
    def aggregate(ratings: Sequence[CaseRating]) -> Dict[str, float]:
        """LLMRating over cases and judges, with per-judge means for auditing."""
        per_case = [r.mean_score for r in ratings if r.mean_score is not None]
        out: Dict[str, float] = {
            "n": len(ratings),
            "llm_rating": mean(per_case) if per_case else 0.0,
            "mean_disagreement": mean(
                [r.disagreement for r in ratings if r.disagreement is not None]
            ) if ratings else 0.0,
        }
        for alias in {a for r in ratings for a in r.scores}:
            scores = [r.scores[alias] for r in ratings
                      if r.scores.get(alias) is not None]
            out[f"judge:{alias}"] = mean(scores) if scores else 0.0
            out[f"judge:{alias}:answered"] = len(scores)
        return out

    @staticmethod
    def save(ratings: Sequence[CaseRating], path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([{
            "case_id": r.case_id, "task": r.task,
            "scores": r.scores, "reasons": r.reasons,
            "mean": r.mean_score, "disagreement": r.disagreement,
        } for r in ratings], indent=2))


def _parse_score(reply: str) -> Optional[int]:
    match = re.search(r"SCORE:\s*([1-5])", reply)
    if match:
        return int(match.group(1))
    # Fall back to the first bare 1-5 in the reply.
    loose = re.search(r"\b([1-5])\b", reply)
    return int(loose.group(1)) if loose else None


def _parse_reason(reply: str) -> str:
    match = re.search(r"REASON:\s*(.+)", reply)
    return match.group(1).strip() if match else reply.strip()[:200]
