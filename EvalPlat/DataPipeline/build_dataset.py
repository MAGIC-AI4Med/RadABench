"""Synthetic-layer construction pipeline (paper Appendix ``sec:construction_pipeline``).

Four stages turn generation output into the evaluation set:

    1. parse_cases   raw <Case> XML  ->  structured per-case JSON
    2. generate_qa   each record     ->  11 task-specific QA pairs
    3. combine_qa    records + QA    ->  the evaluation file
    4. export_xlsx   evaluation file ->  spreadsheet for manual verification

Stage 2 is the one that costs money and time; stages 1, 3 and 4 are pure data
transformations and are safe to re-run.

The generation prompt lives in qa_generation_prompt.txt rather than inline. It
defines the ten-tool toolkit and the eleven task templates, and is what makes
the 24,200 QA pairs reproducible; paraphrasing it produces a different dataset.

Run stages individually::

    python DataPipeline/build_dataset.py parse-cases  --raw Cases/ --out Processed/
    python DataPipeline/build_dataset.py generate-qa  --cases Processed/ --out QA/ --model gpt-4
    python DataPipeline/build_dataset.py combine-qa   --cases Processed/ --qa QA/ --out Data/combined.json
    python DataPipeline/build_dataset.py export-xlsx  --data Data/combined.json --out Data/combined.xlsx

Or the whole thing::

    python DataPipeline/build_dataset.py all --raw Cases/ --workdir build/ --model gpt-4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from Utils.llm_client import LLMClient, available_models  # noqa: E402

PROMPT_PATH = Path(__file__).with_name("qa_generation_prompt.txt")

# The eleven task types, keyed by the tool-category chain each requires. The
# order is fixed: it is what the generation prompt asks for and what the QA
# parser assigns answers to.
TASK_KEYS = ["012", "013", "014", "0123", "01235", "0126", "0136", "01348",
             "0123568", "01235678", "012356789"]

# 22 anatomy-modality combinations x 100 records.
RECORDS_PER_COMBINATION = 100
COMBINATIONS = 22


# --------------------------------------------------------------------------- #
# Stage 1 - parse raw generation output into structured records
# --------------------------------------------------------------------------- #

def parse_case(text: str) -> Dict[str, Union[str, Dict]]:
    # 移除 <Case></Case> 外的内容
    case_content = re.search(r'<Case>(.*?)</Case>', text, re.DOTALL)
    if not case_content:
        return {}
    
    case_content = case_content.group(1).strip()
    
    def parse_element(content: str) -> Union[str, Dict]:
        # 如果内容不包含子元素，直接返回字符串
        if '<' not in content:
            return content.strip()
        
        result = {}
        # 使用正则表达式匹配所有顶层元素
        elements = re.findall(r'<(\w+)>(.*?)</\1>', content, re.DOTALL)
        
        for tag, inner_content in elements:
            # 递归解析子元素
            result[tag] = parse_element(inner_content)
        
        return result
    
    return {"case": parse_element(case_content)}

def parse_cases(raw_dir: Path, out_dir: Path) -> int:
    """Parse every raw <Case> file into structured JSON.

    Files are ordered by the numeric prefix in their name, which fixes a
    record's index and therefore its anatomy-modality combination. Sorting
    lexically would shuffle the dataset.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(os.listdir(raw_dir), key=lambda name: int(name.split("_")[0]))

    written = 0
    for name in tqdm(files, desc="Parsing cases"):
        payload = json.loads((raw_dir / name).read_text())
        parsed = parse_case(payload if isinstance(payload, str) else json.dumps(payload))
        if not parsed:
            print(f"  skipped {name}: no <Case> block found")
            continue
        (out_dir / name).write_text(json.dumps(parsed, indent=4))
        written += 1

    print(f"Parsed {written} of {len(files)} raw files into {out_dir}")
    return written


def combine_records(processed_dir: Path, out_path: Path) -> Dict:
    """Merge per-case files into one index-keyed record file."""
    files = sorted(os.listdir(processed_dir), key=lambda name: int(name.split("_")[0]))
    combined = {
        str(index): json.loads((processed_dir / name).read_text())
        for index, name in enumerate(files)
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=4))
    print(f"Combined {len(combined)} records into {out_path}")
    return combined


# --------------------------------------------------------------------------- #
# Stage 2 - generate the QA pairs
# --------------------------------------------------------------------------- #

def subset_indices(per_combination: int = 5) -> List[int]:
    """The evenly-spread subset: the first N records of each combination.

    Taking the first N of each block keeps every anatomy-modality combination
    represented, which a flat random sample over 2,200 records would not.
    """
    indices: List[int] = []
    for start in range(0, COMBINATIONS * RECORDS_PER_COMBINATION, RECORDS_PER_COMBINATION):
        indices.extend(range(start, start + per_combination))
    return indices


def generate_qa(cases_path: Path, out_dir: Path, model: str,
                indices: List[int] = None) -> int:
    """Ask the model for 11 QA pairs per record.

    Already-generated records are skipped, so an interrupted run resumes rather
    than paying for the same records twice.
    """
    prompt = PROMPT_PATH.read_text()
    records = json.loads(cases_path.read_text())
    keys = list(records)
    targets = indices if indices is not None else range(len(keys))

    llm = LLMClient(model)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for index in tqdm(list(targets), desc="Generating QA"):
        if index >= len(keys):
            continue
        key = keys[index]
        destination = out_dir / f"{key}.json"
        if destination.exists():
            continue

        reply = llm.chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(records[key], ensure_ascii=False)},
        ])
        destination.write_text(json.dumps(reply, ensure_ascii=False))
        generated += 1

    print(f"Generated QA for {generated} records into {out_dir}")
    return generated


# --------------------------------------------------------------------------- #
# Stage 3 - parse QA text and attach it to the records
# --------------------------------------------------------------------------- #

import re

def process_qa_text(text):
    # 首先尝试完整的QA对匹配
    qa_pairs = re.findall(r'<Q(\d+)>(.*?)</Q\1>\s*<A\1>(.*?)</A\1>', text, re.DOTALL)
    
    # 定义键的顺序
    key_order = ['012', '013', '014', '0123', '01235', '0126', '0136', '01348', 
                 '0123568', '01235678', '012356789']
    
    # 创建结果字典
    result = {}
    
    # 如果完整匹配失败，尝试简单匹配
    if not qa_pairs:
        # 使用正则匹配<Q>和<A>之间的内容，直到下一个<或换行符
        pattern = r'<([QA])(\d+)>(.*?)(?=<|$|\n)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        # 整理匹配结果为QA对
        temp_dict = {}
        for tag_type, num, content in matches:
            if num not in temp_dict:
                temp_dict[num] = {}
            temp_dict[num][tag_type] = content.strip()
        
        # 转换为qa_pairs格式
        qa_pairs = [(num, temp_dict[num]['Q'], temp_dict[num]['A']) 
                   for num in temp_dict 
                   if 'Q' in temp_dict[num] and 'A' in temp_dict[num]]
    
    # 处理QA对
    for i, (num, q, a) in enumerate(qa_pairs):
        if i < len(key_order):
            key = key_order[i]
            result[key] = {
                'Q': q.strip(),
                'A': a.strip()
            }
    
    return result

# 你的输入文本

def combine_qa(cases_path: Path, qa_dir: Path, out_path: Path,
               indices: List[int] = None, require_complete: bool = False) -> Dict:
    """Attach parsed QA pairs to their records.

    require_complete drops records with fewer than 11 task types. It defaults
    to False because the published subset itself contains six such records
    (1,204 QA pairs rather than 1,210), so dropping them would not reproduce
    the shipped file. The count is reported either way, since an uneven
    per-task sample size matters before it reaches a results table.
    """
    records = json.loads(cases_path.read_text())
    keys = list(records)
    targets = indices if indices is not None else range(len(keys))

    combined, incomplete = {}, 0
    for index in tqdm(list(targets), desc="Combining QA"):
        if index >= len(keys):
            continue
        key = keys[index]
        qa_file = qa_dir / f"{key}.json"
        if not qa_file.exists():
            continue

        parsed = process_qa_text(json.loads(qa_file.read_text()))
        if len(parsed) < len(TASK_KEYS):
            incomplete += 1
            if require_complete:
                continue

        record = dict(records[key])
        record["QA"] = parsed
        combined[key] = record

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=4))

    pairs = sum(len(r["QA"]) for r in combined.values())
    print(f"Wrote {len(combined)} records / {pairs} QA pairs to {out_path}")
    if incomplete:
        verb = "Dropped" if require_complete else "Kept"
        print(f"{verb} {incomplete} records with fewer than "
              f"{len(TASK_KEYS)} task types "
              f"(pass --require-complete to drop them)")
    return combined


# --------------------------------------------------------------------------- #
# Stage 4 - export for manual verification
# --------------------------------------------------------------------------- #

def export_xlsx(data_path: Path, out_path: Path) -> int:
    """Flatten to one row per QA pair, for the clinician review pass."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to export a spreadsheet.") from exc

    records = json.loads(data_path.read_text())
    rows = [
        {"Case ID": case_id, "Tool Chain": chain,
         "Question": qa.get("Q", ""), "Answer": qa.get("A", "")}
        for case_id, record in records.items()
        for chain, qa in record.get("QA", {}).items()
    ]
    pd.DataFrame(rows).to_excel(out_path, index=False)
    print(f"Wrote {len(rows)} QA rows to {out_path}")
    return len(rows)


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the RadA-BenchPlat synthetic layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p1 = sub.add_parser("parse-cases", help="raw <Case> XML -> structured JSON")
    p1.add_argument("--raw", required=True, type=Path)
    p1.add_argument("--out", required=True, type=Path)
    p1.add_argument("--combined", type=Path, default=None,
                    help="also write the merged record file here")

    p2 = sub.add_parser("generate-qa", help="records -> 11 QA pairs each")
    p2.add_argument("--cases", required=True, type=Path)
    p2.add_argument("--out", required=True, type=Path)
    p2.add_argument("--model", default="gpt-4", help=", ".join(available_models()))
    p2.add_argument("--subset", type=int, default=None,
                    help="records per anatomy-modality combination (default: all)")

    p3 = sub.add_parser("combine-qa", help="records + QA -> evaluation file")
    p3.add_argument("--cases", required=True, type=Path)
    p3.add_argument("--qa", required=True, type=Path)
    p3.add_argument("--out", required=True, type=Path)
    p3.add_argument("--subset", type=int, default=None)
    p3.add_argument("--require-complete", action="store_true",
                    help="drop records with fewer than 11 task types")

    p4 = sub.add_parser("export-xlsx", help="evaluation file -> spreadsheet")
    p4.add_argument("--data", required=True, type=Path)
    p4.add_argument("--out", required=True, type=Path)

    p5 = sub.add_parser("all", help="run every stage in order")
    p5.add_argument("--raw", required=True, type=Path)
    p5.add_argument("--workdir", required=True, type=Path)
    p5.add_argument("--model", default="gpt-4")
    p5.add_argument("--subset", type=int, default=None)
    p5.add_argument("--require-complete", action="store_true")

    args = parser.parse_args()

    if args.stage == "parse-cases":
        parse_cases(args.raw, args.out)
        if args.combined:
            combine_records(args.out, args.combined)

    elif args.stage == "generate-qa":
        generate_qa(args.cases, args.out, args.model,
                    subset_indices(args.subset) if args.subset else None)

    elif args.stage == "combine-qa":
        combine_qa(args.cases, args.qa, args.out,
                   subset_indices(args.subset) if args.subset else None,
                   require_complete=args.require_complete)

    elif args.stage == "export-xlsx":
        export_xlsx(args.data, args.out)

    elif args.stage == "all":
        work = args.workdir
        processed, qa_dir = work / "processed", work / "qa"
        combined, final = work / "records.json", work / "evaluation.json"
        indices = subset_indices(args.subset) if args.subset else None

        parse_cases(args.raw, processed)
        combine_records(processed, combined)
        generate_qa(combined, qa_dir, args.model, indices)
        combine_qa(combined, qa_dir, final, indices,
                   require_complete=args.require_complete)
        export_xlsx(final, work / "evaluation.xlsx")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
