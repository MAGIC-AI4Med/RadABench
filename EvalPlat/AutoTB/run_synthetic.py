"""Score synthetic-layer AutoTB over a directory of agent transcripts.

    python AutoTB/run_synthetic.py \
        --logs Outputs/claude-sonnet-4.5/multiDenyl2/logouts \
        --condition config2 \
        --data Data/combined_casesV3_qa_Subset.json

Transcripts are the per-case logs written by the EvalPipeline scripts. The case
id comes from the file name, and the tool-chain signature in the name selects
which category that case removed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AutoTB import run_synthetic_autotb, summarize  # noqa: E402

# Tool-chain signature -> the category removed for it under config1/config3.
# Same mapping as EvalMetrics/multihop_deny3.py.
DENY_MAP = {
    "012": "Organ Segmentor",       "013": "Anomaly Detector",
    "014": "Disease Diagnoser",     "0123": "Organ Segmentor",
    "01235": "Disease Diagnoser",   "0126": "Biomarker Quantifier",
    "0136": "Biomarker Quantifier", "01348": "Anomaly Detector",
    "0123568": "Disease Diagnoser", "01235678": "Indicator Evaluator",
    "012356789": "Indicator Evaluator",
}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic-layer AutoTB scoring (paper sec:autotb_protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--logs", required=True, help="directory of agent transcripts")
    parser.add_argument("--condition", required=True, choices=["config1", "config2", "config3"])
    parser.add_argument("--data", default="Data/combined_casesV3_qa_Subset.json")
    parser.add_argument("--out", default=None, help="write per-case results as JSON")
    args = parser.parse_args()

    with open(args.data) as handle:
        cases = json.load(handle)
    case_list = list(cases.values())

    log_dir = Path(args.logs)
    if not log_dir.is_dir():
        parser.error(f"--logs directory not found: {log_dir}")

    results = []
    for path in sorted(log_dir.rglob("*.txt"), key=lambda p: natural_key(str(p))):
        signature = path.stem
        # Logs live at logouts/<case index>/<signature>.txt
        try:
            case_index = int(path.parent.name)
        except ValueError:
            continue
        if case_index >= len(case_list):
            continue

        results.append(run_synthetic_autotb(
            case_id=f"{case_index}/{signature}",
            transcript=path.read_text(errors="replace"),
            condition=args.condition,
            case=case_list[case_index],
            target_category=DENY_MAP.get(signature, ""),
        ))

    if not results:
        print(f"No transcripts found under {log_dir}", file=sys.stderr)
        return 1

    stats = summarize(results)
    print(f"Condition           : {args.condition}")
    print(f"Cases scored        : {stats['n']}")
    print(f"AutoTB trigger rate : {stats['trigger_rate']:.3f}")
    print(f"Grounding rate      : {stats['grounding_rate']:.3f}")
    print(f"AutoTB success rate : {stats['execution_success_rate']:.3f}")
    print("Failure modes       :")
    for mode, count in stats["failure_modes"].items():
        if count:
            print(f"    {mode:<22} {count:>4}  ({count / stats['n']:.1%})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"summary": stats,
             "cases": [r.__dict__ for r in results]}, indent=2, default=str))
        print(f"\nPer-case results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
