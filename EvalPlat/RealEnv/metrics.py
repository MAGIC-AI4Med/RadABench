"""Task-specific metrics for the real-environment layer (sec:eval_metrics).

    organ / anomaly joint grounding   Dice, Jaccard
    diagnosis with grounding clues    Accuracy, Jaccard
    standard report generation        RadGraph-F1, RaTE Score

All three also carry LLM Rating and Human Rating, which need a model call or a
reviewer file rather than an array; see llm_rating.py.

Dice, Jaccard and Accuracy are implemented here, since they are simple and a
package dependency would make results drift with a library version. RadGraph-F1
and RaTE wrap their reference implementations instead of being approximated: an
approximation would produce numbers not comparable with any published figure.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

__all__ = ["dice", "jaccard", "set_jaccard", "accuracy", "radgraph_f1",
           "rate_score", "reported_ceiling_gap", "summarize_task"]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- #
# grounding: spatial overlap
# --------------------------------------------------------------------------- #

def _as_bool_array(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("numpy is required for mask metrics.") from exc
    return np.asarray(mask).astype(bool)


def dice(predicted, reference) -> float:
    """Dice = 2|M and M*| / (|M| + |M*|).

    Two empty masks score 1.0: predicting that nothing is there is a correct
    answer, not a division by zero.
    """
    pred, ref = _as_bool_array(predicted), _as_bool_array(reference)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {ref.shape}")
    total = pred.sum() + ref.sum()
    if total == 0:
        return 1.0
    return float(2 * (pred & ref).sum() / total)


def jaccard(predicted, reference) -> float:
    """Jaccard = |M and M*| / |M or M*|, same empty-mask convention as Dice."""
    pred, ref = _as_bool_array(predicted), _as_bool_array(reference)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {ref.shape}")
    union = (pred | ref).sum()
    if union == 0:
        return 1.0
    return float((pred & ref).sum() / union)


# --------------------------------------------------------------------------- #
# diagnosis: label agreement
# --------------------------------------------------------------------------- #

def accuracy(predicted: Sequence[str], reference: Sequence[str]) -> float:
    """Exact-match accuracy over per-case diagnoses."""
    if len(predicted) != len(reference):
        raise ValueError("predicted and reference must be the same length")
    if not predicted:
        return 0.0
    return _mean(int(p == r) for p, r in zip(predicted, reference))


def set_jaccard(predicted: Set[str] | Sequence[str],
                reference: Set[str] | Sequence[str]) -> float:
    """Jaccard over diagnosis sets, for cases with multiple findings.

    Separate from the mask version above: diagnosis is scored on label overlap,
    grounding on pixel overlap. Both are reported as "Jaccard Index", so the two
    functions are kept apart.
    """
    pred, ref = set(predicted), set(reference)
    if not pred and not ref:
        return 1.0
    union = pred | ref
    return len(pred & ref) / len(union) if union else 0.0


# --------------------------------------------------------------------------- #
# report generation: clinical text quality
# --------------------------------------------------------------------------- #

def radgraph_f1(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """RadGraph-F1 over extracted clinical entities and relations.

    Wraps the reference implementation:

        pip install radgraph

    Not approximated: RadGraph-F1 depends on a specific entity and relation
    extractor, and a lexical stand-in would give plausible-looking numbers that
    cannot be compared with published results.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    try:
        from radgraph import F1RadGraph
    except ImportError as exc:
        raise ImportError(
            "RadGraph-F1 requires the `radgraph` package: pip install radgraph\n"
            "It is not approximated here because an approximation would not be "
            "comparable with published RadGraph-F1 scores."
        ) from exc

    scorer = F1RadGraph(reward_level="partial")
    score, _, _, _ = scorer(hyps=list(hypotheses), refs=list(references))
    return float(score)


def rate_score(hypotheses: Sequence[str], references: Sequence[str],
               model_path: str = "Angelakeke/RaTE-NER-Deberta") -> float:
    """RaTE Score: radiology-oriented alignment with the reference report.

    Wraps the reference implementation:

        pip install RaTEScore
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    try:
        from RaTEScore import RaTEScore
    except ImportError as exc:
        raise ImportError(
            "RaTE Score requires the `RaTEScore` package: pip install RaTEScore"
        ) from exc

    scorer = RaTEScore()
    return float(_mean(scorer.compute_score(list(hypotheses), list(references))))


# --------------------------------------------------------------------------- #
# the idealized-executable gap
# --------------------------------------------------------------------------- #

def reported_ceiling_gap(ceiling: float, current_best: float) -> float:
    """Delta = (U - C) / C * 100%, the idealized-vs-executable gap.

    ceiling substitutes a tool's officially reported performance into the
    idealized protocol. current_best is the column-wise best observed score
    across evaluated systems.

    Neither is what one deployable agent achieves: U assumes perfect
    orchestration and C is an envelope across systems. Treat it as a
    computational reference point, not headroom.
    """
    if current_best == 0:
        raise ValueError("current_best must be non-zero to compute a relative gap")
    return (ceiling - current_best) / current_best * 100.0


# --------------------------------------------------------------------------- #
# per-task rollup
# --------------------------------------------------------------------------- #

def summarize_task(task: str, records: List[Dict]) -> Dict[str, float]:
    """Score one real-environment task from its per-case records.

    Expected keys by task:

        organ_anomaly_grounding   predicted_mask, reference_mask
        diagnosis_with_clues      predicted, reference (strings or sets)
        report_generation         hypothesis, reference (strings)

    Any task may also carry llm_rating and human_rating.
    """
    if not records:
        return {"n": 0}

    out: Dict[str, float] = {"n": len(records)}

    if task == "organ_anomaly_grounding":
        pairs = [(r["predicted_mask"], r["reference_mask"]) for r in records
                 if "predicted_mask" in r and "reference_mask" in r]
        out["dice"] = _mean(dice(p, g) for p, g in pairs)
        out["jaccard"] = _mean(jaccard(p, g) for p, g in pairs)

    elif task == "diagnosis_with_clues":
        pairs = [(r["predicted"], r["reference"]) for r in records
                 if "predicted" in r and "reference" in r]
        out["accuracy"] = accuracy([p for p, _ in pairs], [g for _, g in pairs])
        out["jaccard"] = _mean(
            set_jaccard(p if isinstance(p, (set, list)) else {p},
                        g if isinstance(g, (set, list)) else {g})
            for p, g in pairs
        )

    elif task == "report_generation":
        hyps = [r["hypothesis"] for r in records if "hypothesis" in r]
        refs = [r["reference"] for r in records if "reference" in r]
        # Let the ImportError surface. A skipped metric is worse than a run
        # that stops and names the missing package.
        out["radgraph_f1"] = radgraph_f1(hyps, refs)
        out["rate_score"] = rate_score(hyps, refs)

    else:
        raise ValueError(
            f"Unknown task {task!r}. Expected one of: organ_anomaly_grounding, "
            f"diagnosis_with_clues, report_generation."
        )

    if any("llm_rating" in r for r in records):
        out["llm_rating"] = _mean(r["llm_rating"] for r in records if "llm_rating" in r)
    if any("human_rating" in r for r in records):
        out["human_rating"] = _mean(r["human_rating"] for r in records if "human_rating" in r)

    return out
