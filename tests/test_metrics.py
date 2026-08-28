"""Unit tests for the agent-level metrics.

Each test pins one formula from Appendix ``sec:eval_metrics`` with a case
worked by hand, so a refactor that changes a denominator fails here instead
of shifting a published number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EvalPlat"))

from EvalMetrics.metrics import (  # noqa: E402
    aggregate, bleu, chain_distances, execution_completion_rate,
    false_discovery_rate, levenshtein_distance, milestone_hit_rate,
    optimal_tool_score, pre_failure_success_percentage, rouge,
    target_hit_rate, task_completion_rate, tool_matching_accuracy,
    unsolvability_awareness_rate, unsolvability_grounding_rate, word_f1,
)

GT = ["AC", "MC", "OS", "BQ"]          # organ-wise biomarker calculation
PASS = FAIL = 0


def check(label, got, want, tol=1e-9):
    global PASS, FAIL
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    print(f"  {'✓' if ok else '✗'} {label:<52} got={got!r:<22} want={want!r}")
    PASS, FAIL = PASS + ok, FAIL + (not ok)


print("── Levenshtein Distance ──")
check("identical chains", levenshtein_distance(GT, GT), 0)
check("one extra tool", levenshtein_distance(["AC", "MC", "OS", "RG", "BQ"], GT), 1)
check("one missing tool", levenshtein_distance(["AC", "MC", "BQ"], GT), 1)
check("one substitution", levenshtein_distance(["AC", "MC", "AD", "BQ"], GT), 1)
check("empty vs full", levenshtein_distance([], GT), 4)

print("\n── chain_distances: planned well, executed badly ──")
d = chain_distances(planned=GT, executed=["AC", "MC", "AD"], ground_truth=GT)
check("plan_dist (agent planned correctly)", d["plan_dist"], 0)
check("exec_dist (execution drifted)", d["exec_dist"], 2)
check("cons_dist (plan vs execution)", d["cons_dist"], 2)

print("\n── False Discovery Rate ──")
check("no extra tools", false_discovery_rate(GT, GT), 0.0)
check("1 of 5 not required", false_discovery_rate(["AC", "MC", "OS", "BQ", "TP"], GT), 1 / 5)
check("empty chain guarded by max(|X|,1)", false_discovery_rate([], GT), 0.0)

print("\n── Tool Matching Accuracy ──")
check("exact match", tool_matching_accuracy(GT, GT), 1.0)
check("stops after 2 of 4 steps", tool_matching_accuracy(["AC", "MC"], GT), 0.5)
check("right tools, wrong order", tool_matching_accuracy(["MC", "AC", "OS", "BQ"], GT), 0.5)

print("\n── Optimal Tool Score ──")
check("always picked the best", optimal_tool_score([1, 1, 1], [3, 3, 3]), 1.0)
check("always picked the worst", optimal_tool_score([3, 3, 3], [3, 3, 3]), 1 / 3)
check("single candidate is unscoreable -> 1.0", optimal_tool_score([1], [1]), 1.0)
check("(N-R+1)/N worked by hand", optimal_tool_score([2], [4]), 0.75)

print("\n── ECR / PFSP ──")
steps, lengths = [4, 2, 4, 1], [4, 4, 4, 4]
check("ECR: 2 of 4 chains completed", execution_completion_rate(steps, lengths), 0.5)
check("PFSP: mean of 2/4 and 1/4 over failures only",
      pre_failure_success_percentage(steps, lengths), (0.5 + 0.25) / 2)
check("PFSP is 0.0 when nothing failed",
      pre_failure_success_percentage([4, 4], [4, 4]), 0.0)
check("PFSP ignores successes (would be 0.4375 if averaged in)",
      pre_failure_success_percentage(steps, lengths) != 0.4375, True)

print("\n── Response metrics ──")
check("TCR", task_completion_rate([True, False, True, True]), 0.75)
check("THR: type mismatch scores 0",
      target_hit_rate(["Report", "Mask"], ["Report", "Report"]), 0.5)
check("MHR", milestone_hit_rate([True, True, False]), 2 / 3)
check("BLEU identical text", bleu("the lung is clear", "the lung is clear"), 1.0)
check("BLEU disjoint text", bleu("aaa bbb ccc ddd", "eee fff ggg hhh"), 0.0)
check("ROUGE-1 recall, 3 of 4 reference unigrams",
      rouge("the lung is", "the lung is clear"), 0.75)
check("word F1 identical", word_f1("a b c", "a b c"), 1.0)

print("\n── Unsolvability: awareness outruns grounding ──")
refused = [True, True, True, False]
grounded = [True, False, False, False]
check("UAR", unsolvability_awareness_rate(refused), 0.75)
check("UGR", unsolvability_grounding_rate(grounded), 0.25)
check("UAR > UGR, the paper's reported pattern",
      unsolvability_awareness_rate(refused) > unsolvability_grounding_rate(grounded), True)

print("\n── aggregate() skips metrics it lacks inputs for ──")
partial = aggregate([{"executed": GT, "ground_truth": GT}])
check("chain metrics present", "tma" in partial, True)
check("ECR absent without steps_done", "ecr" not in partial, True)
check("aggregate on empty input", aggregate([]), {})

print(f"\n{'='*72}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
