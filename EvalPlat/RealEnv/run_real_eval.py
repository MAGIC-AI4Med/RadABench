"""Run the real-environment evaluation over the 165-case manifest.

    python RealEnv/run_real_eval.py \
        --model claude-sonnet-4.5 \
        --manifest RealEnv/manifest.json \
        --tool-pool RealEnv/tool_pool.json \
        --task report_generation \
        --out Outputs/real

Produces per-case traces, the task-specific automatic metrics, and with
--llm-rating the three-judge clinical score. Human ratings are merged from a
reviewer file via --human-ratings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RealEnv.agent import RealAgent  # noqa: E402
from RealEnv.cases import TASKS, CaseManifest  # noqa: E402
from RealEnv.llm_rating import LLMRater  # noqa: E402
from RealEnv.metrics import summarize_task  # noqa: E402
from RealEnv.tool_pool import ExecutableToolPool, build_default_pool  # noqa: E402
from Utils.llm_client import LLMClient, available_models  # noqa: E402


def load_executor(spec: str):
    """Resolve module:function into the callable that runs a tool.

    Injected rather than hard-wired, since the executable tools are M3Builder
    checkpoints, nnU-Net and RP3D-DM, which live outside this repository and
    differ per installation. A stub executor returning plausible arrays would
    produce numbers that look like results and are not.
    """
    if not spec:
        raise SystemExit(
            "--executor is required. Point it at the function that invokes your\n"
            "installed tools, e.g. --executor mytools.runner:run_tool\n"
            "It receives (ExecutableTool, dict_of_inputs) and returns a dict of\n"
            "outputs keyed like the tool card's `output` field."
        )
    module_name, _, function_name = spec.partition(":")
    if not function_name:
        raise SystemExit(f"--executor must be module:function, got {spec!r}")
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-environment evaluation (paper §3.3, Table 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="claude-sonnet-4.5",
                        help="agent core: " + ", ".join(available_models()))
    parser.add_argument("--manifest", default="RealEnv/manifest.json")
    parser.add_argument("--tool-pool", default=None,
                        help="tool pool JSON; defaults to the built-in 42-variant pool")
    parser.add_argument("--executor", default="",
                        help="module:function that actually runs a tool")
    parser.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    parser.add_argument("--out", default="Outputs/real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm-rating", action="store_true",
                        help="score answers with the three-judge panel")
    parser.add_argument("--human-ratings", default=None,
                        help="JSON of {case_id: score} from blinded expert review")
    parser.add_argument("--skip-validation", action="store_true",
                        help="allow a manifest that is not the full 165-case set")
    args = parser.parse_args()

    manifest = CaseManifest.load(args.manifest, validate=not args.skip_validation)
    pool = (ExecutableToolPool.from_json(args.tool_pool)
            if args.tool_pool else build_default_pool())
    executor = load_executor(args.executor)
    agent = RealAgent(LLMClient(args.model), pool, executor)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    out_root = Path(args.out) / args.model
    out_root.mkdir(parents=True, exist_ok=True)

    human = {}
    if args.human_ratings:
        human = json.loads(Path(args.human_ratings).read_text())

    overall = {}
    for task in tasks:
        cases = manifest.by_task(task)
        if args.limit:
            cases = cases[:args.limit]
        print(f"\n=== {task} ({len(cases)} cases) ===")

        runs, records = [], []
        for case in cases:
            run = agent.run(case)
            runs.append(run)
            record = {"case_id": case.case_id, **_reference_fields(task, case, run)}
            if case.case_id in human:
                record["human_rating"] = human[case.case_id]
            records.append(record)
            flag = "refused" if run.refused else ("error" if run.error else "ok")
            print(f"  {case.case_id:<28} steps={run.steps_done:<3} {flag}")

        if args.llm_rating:
            rater = LLMRater()
            ratings = rater.rate_all([{
                "case_id": c.case_id, "task": task, "question": c.question,
                "answer": r.answer, "reference": json.dumps(c.reference, default=str),
            } for c, r in zip(cases, runs)])
            rater.save(ratings, str(out_root / f"{task}_llm_ratings.json"))
            for record, rating in zip(records, ratings):
                if rating.mean_score is not None:
                    record["llm_rating"] = rating.mean_score
            print(f"  LLM rating: {LLMRater.aggregate(ratings)['llm_rating']:.3f}")

        (out_root / f"{task}_runs.json").write_text(
            json.dumps([r.as_dict() for r in runs], indent=2, default=str))

        try:
            summary = summarize_task(task, records)
        except ImportError as error:
            summary = {"n": len(records), "error": str(error)}
        overall[task] = summary
        print(f"  {json.dumps(summary, default=str)}")

    (out_root / "summary.json").write_text(json.dumps(overall, indent=2, default=str))
    print(f"\nWritten to {out_root}")
    return 0


def _reference_fields(task, case, run):
    """Map a run's answer onto the keys the task's metrics expect."""
    if task == "organ_anomaly_grounding":
        return {"predicted_mask": run.steps[-1].outputs.get("$AnomalyDim$")
                if run.steps else None,
                "reference_mask": case.reference.get("lesion_annotation")}
    if task == "diagnosis_with_clues":
        return {"predicted": run.answer.strip(),
                "reference": case.reference.get("diagnosis", "")}
    return {"hypothesis": run.answer,
            "reference": " ".join(filter(None, [case.reference.get("finding", ""),
                                                case.reference.get("impression", "")]))}


if __name__ == "__main__":
    raise SystemExit(main())
