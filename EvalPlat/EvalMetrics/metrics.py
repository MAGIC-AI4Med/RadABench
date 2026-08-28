"""Agent-level metrics, following Appendix sec:eval_metrics.

Each function implements one formula and takes plain Python data (tool-category
lists, counts, strings), so it can be tested without an agent or a log file.
Log parsing stays in the per-condition scripts.

Metric names follow the paper: LD, FDR, TMA, OTS, ECR, PFSP, TCR, THR, MHR,
BLEU/ROUGE/F1, UAR, UGR.

Real-case metrics are in RealEnv/metrics.py, AutoTB metrics in
AutoTB/protocol.py.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence

__all__ = [
    "levenshtein_distance", "chain_distances", "false_discovery_rate",
    "tool_matching_accuracy", "optimal_tool_score",
    "execution_completion_rate", "pre_failure_success_percentage",
    "task_completion_rate", "target_hit_rate", "milestone_hit_rate",
    "bleu", "rouge", "word_f1",
    "unsolvability_awareness_rate", "unsolvability_grounding_rate",
    "aggregate",
]

Chain = Sequence[str]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- #
# Chain planning and execution consistency
# --------------------------------------------------------------------------- #

def levenshtein_distance(x: Chain, g: Chain) -> int:
    """LD(X, G): edit distance between two tool-category chains.

    Insert / delete / substitute each cost 1 (c(o) = 1). Works on category
    sequences, not characters.
    """
    if not x:
        return len(g)
    if not g:
        return len(x)

    previous = list(range(len(g) + 1))
    for i, xi in enumerate(x, start=1):
        current = [i]
        for j, gj in enumerate(g, start=1):
            current.append(min(
                previous[j] + 1,                        # deletion
                current[j - 1] + 1,                     # insertion
                previous[j - 1] + (xi != gj),           # substitution
            ))
        previous = current
    return previous[-1]


def chain_distances(planned: Chain, executed: Chain, ground_truth: Chain) -> Dict[str, int]:
    """Plan Dist., Exec. Dist. and Cons. Dist. together.

    A small plan_dist with a large exec_dist means the agent planned correctly
    and drifted during execution.
    """
    return {
        "plan_dist": levenshtein_distance(planned, ground_truth),
        "exec_dist": levenshtein_distance(executed, ground_truth),
        "cons_dist": levenshtein_distance(planned, executed),
    }


def false_discovery_rate(x: Chain, g: Chain) -> float:
    """FDR(X, G): share of called tools not required by the ground truth.

    max(|X|, 1) keeps an empty chain at 0.0 instead of undefined.
    """
    required = set(g)
    unnecessary = sum(1 for item in x if item not in required)
    return unnecessary / max(len(x), 1)


def tool_matching_accuracy(x: Chain, g: Chain) -> float:
    """TMA(X, G): order-sensitive agreement, position by position.

    Normalised by ground-truth length L, so a chain that stops early is
    penalised for the steps it never reached.
    """
    length = len(g)
    if length == 0:
        return 0.0
    hits = sum(1 for i in range(length) if i < len(x) and x[i] == g[i])
    return hits / length


# --------------------------------------------------------------------------- #
# Optimal orchestrating
# --------------------------------------------------------------------------- #

def optimal_tool_score(ranks: Sequence[int], candidate_counts: Sequence[int]) -> float:
    """OTS: how close each selected tool ranks to the best available one.

    ranks: R_t per step, the performance rank of the chosen tool (1 = best).
    candidate_counts: N_t per step, how many tools could serve that category.

    A step with only one candidate scores 1.0, since there was no choice to
    get wrong.
    """
    if len(ranks) != len(candidate_counts):
        raise ValueError("ranks and candidate_counts must be the same length")
    per_step = [
        (n - r + 1) / n
        for r, n in zip(ranks, candidate_counts)
        if n > 0
    ]
    return _mean(per_step)


# --------------------------------------------------------------------------- #
# I/O organizing
# --------------------------------------------------------------------------- #

def execution_completion_rate(steps_done: Sequence[int], chain_lengths: Sequence[int]) -> float:
    """ECR: fraction of cases whose whole chain ran without failure.

    steps_done[j] counts steps executed before termination. A case counts only
    when it reaches the full ground-truth chain length.
    """
    if len(steps_done) != len(chain_lengths):
        raise ValueError("steps_done and chain_lengths must be the same length")
    if not steps_done:
        return 0.0
    return _mean(int(s >= L) for s, L in zip(steps_done, chain_lengths))


def pre_failure_success_percentage(steps_done: Sequence[int], chain_lengths: Sequence[int]) -> float:
    """PFSP: how far the failed cases got before breaking.

    Computed over failures only. Including successes would erase the
    distinction between failing at step 1 and failing at the last step.

    Returns 0.0 when nothing failed.
    """
    if len(steps_done) != len(chain_lengths):
        raise ValueError("steps_done and chain_lengths must be the same length")
    failed = [(s, L) for s, L in zip(steps_done, chain_lengths) if s < L and L > 0]
    if not failed:
        return 0.0
    return _mean(s / L for s, L in failed)


# --------------------------------------------------------------------------- #
# Response synthesizing
# --------------------------------------------------------------------------- #

def task_completion_rate(completed: Sequence[bool]) -> float:
    """TCR: end-to-end success, workflow completed and task target satisfied."""
    return _mean(int(bool(c)) for c in completed) if completed else 0.0


def target_hit_rate(predicted_types: Sequence[str], reference_types: Sequence[str]) -> float:
    """THR: does the final output type match the required response target?

    A correct chain that returns a mask when the task wanted a report scores 0.
    """
    if len(predicted_types) != len(reference_types):
        raise ValueError("predicted_types and reference_types must be the same length")
    if not predicted_types:
        return 0.0
    return _mean(int(p == r) for p, r in zip(predicted_types, reference_types))


def milestone_hit_rate(milestones_hit: Sequence[bool]) -> float:
    """MHR: was the task's key intermediate result produced?

    Milestones include organ masks, anomaly localisation, biomarker values and
    preliminary diagnoses.
    """
    return _mean(int(bool(m)) for m in milestones_hit) if milestones_hit else 0.0


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """BLEU with clipped n-gram precision and brevity penalty.

    Implemented locally so scoring is identical across environments and needs
    no tokenizer download.
    """
    hyp, ref = hypothesis.split(), reference.split()
    if not hyp or not ref:
        return 0.0

    weights, log_precisions = 1.0 / max_n, []
    for n in range(1, max_n + 1):
        hyp_grams, ref_grams = _ngrams(hyp, n), _ngrams(ref, n)
        total = sum(hyp_grams.values())
        if total == 0:
            return 0.0
        clipped = sum(min(c, ref_grams[g]) for g, c in hyp_grams.items())
        if clipped == 0:
            return 0.0
        log_precisions.append(weights * math.log(clipped / total))

    brevity = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return brevity * math.exp(sum(log_precisions))


def rouge(hypothesis: str, reference: str, n: int = 1) -> float:
    """ROUGE recall: reference n-grams recovered by the hypothesis."""
    hyp, ref = hypothesis.split(), reference.split()
    if not ref:
        return 0.0
    hyp_grams, ref_grams = _ngrams(hyp, n), _ngrams(ref, n)
    if not ref_grams:
        return 0.0
    overlap = sum(min(c, hyp_grams[g]) for g, c in ref_grams.items())
    return overlap / sum(ref_grams.values())


def word_f1(hypothesis: str, reference: str) -> float:
    """Word-overlap F1: harmonic mean of token precision and recall."""
    hyp, ref = Counter(hypothesis.split()), Counter(reference.split())
    overlap = sum((hyp & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hyp.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Unsolvability parsing
# --------------------------------------------------------------------------- #

def unsolvability_awareness_rate(refused: Sequence[bool]) -> float:
    """UAR: did the agent recognise the task could not be solved?"""
    return _mean(int(bool(a)) for a in refused) if refused else 0.0


def unsolvability_grounding_rate(grounded: Sequence[bool]) -> float:
    """UGR: did the agent identify what was missing?

    Report alongside UAR. Agents detect impossibility more reliably than they
    locate its cause, and the gap between the two is a result in itself.
    """
    return _mean(int(bool(u)) for u in grounded) if grounded else 0.0


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def aggregate(cases: List[Dict]) -> Dict[str, float]:
    """Roll per-case records up into the reported table.

    A case dict may carry any of: planned, executed, ground_truth, steps_done,
    chain_length, ranks, candidate_counts, completed, predicted_type,
    reference_type, milestone, refused, grounded, hypothesis, reference.

    Missing keys are skipped, not defaulted, so a partial run reports only the
    metrics it can support instead of scoring zeros.
    """
    if not cases:
        return {}

    out: Dict[str, float] = {"n": len(cases)}

    chains = [c for c in cases if "executed" in c and "ground_truth" in c]
    if chains:
        dists = [chain_distances(c.get("planned", c["executed"]), c["executed"], c["ground_truth"])
                 for c in chains]
        out["plan_dist"] = _mean(d["plan_dist"] for d in dists)
        out["exec_dist"] = _mean(d["exec_dist"] for d in dists)
        out["cons_dist"] = _mean(d["cons_dist"] for d in dists)
        out["fdr"] = _mean(false_discovery_rate(c["executed"], c["ground_truth"]) for c in chains)
        out["tma"] = _mean(tool_matching_accuracy(c["executed"], c["ground_truth"]) for c in chains)

    io_cases = [c for c in cases if "steps_done" in c and "chain_length" in c]
    if io_cases:
        steps = [c["steps_done"] for c in io_cases]
        lengths = [c["chain_length"] for c in io_cases]
        out["ecr"] = execution_completion_rate(steps, lengths)
        out["pfsp"] = pre_failure_success_percentage(steps, lengths)

    ots_cases = [c for c in cases if "ranks" in c and "candidate_counts" in c]
    if ots_cases:
        out["ots"] = _mean(optimal_tool_score(c["ranks"], c["candidate_counts"]) for c in ots_cases)

    if any("completed" in c for c in cases):
        out["tcr"] = task_completion_rate([c["completed"] for c in cases if "completed" in c])
    if any("milestone" in c for c in cases):
        out["mhr"] = milestone_hit_rate([c["milestone"] for c in cases if "milestone" in c])

    thr_cases = [c for c in cases if "predicted_type" in c and "reference_type" in c]
    if thr_cases:
        out["thr"] = target_hit_rate([c["predicted_type"] for c in thr_cases],
                                     [c["reference_type"] for c in thr_cases])

    if any("refused" in c for c in cases):
        out["uar"] = unsolvability_awareness_rate([c["refused"] for c in cases if "refused" in c])
    if any("grounded" in c for c in cases):
        out["ugr"] = unsolvability_grounding_rate([c["grounded"] for c in cases if "grounded" in c])

    text_cases = [c for c in cases if c.get("hypothesis") and c.get("reference")]
    if text_cases:
        out["bleu"] = _mean(bleu(c["hypothesis"], c["reference"]) for c in text_cases)
        out["rouge"] = _mean(rouge(c["hypothesis"], c["reference"]) for c in text_cases)
        out["f1"] = _mean(word_f1(c["hypothesis"], c["reference"]) for c in text_cases)

    return out
