"""Construct the 165-case manifest from local copies of the source datasets.

Neither source can be redistributed here: Chest Imagenome derives from
MIMIC-CXR and RadGenome-BrainMRI from the BraTS cohorts, both credentialed.
This script applies the paper's documented selection criteria to your local
copies so the evaluation set can be rebuilt exactly.

    python RealEnv/build_manifest.py \
        --chest-imagenome /data/chest-imagenome \
        --radgenome /data/radgenome-brainmri \
        --out RealEnv/manifest.json

Selection criteria (paper ``sec:real_env_data``):

**Chest Imagenome → organ/anomaly joint grounding, 55 cases.**
Keep cases with an accessible chest radiograph and non-empty local anatomical
*and* abnormality annotations. Map the scene-graph regions and observation
labels onto the benchmark schema (anatomy = chest, modality = X-ray). Exclude
a case when the target region is ambiguous, the abnormality label does not map,
or the image/annotation pair is incomplete. Sample 55 with diverse anatomical
targets and abnormality types.

**RadGenome-BrainMRI → 110 cases, split 55 diagnosis / 55 report generation.**
Sample from BraTS-GLI, BraTS-MEN and BraTS-MET (glioma, meningioma,
metastasis), balanced across cohorts as far as availability allows. Require a
complete image file, usable lesion annotation, and an unambiguous task target.
All are treated as head-and-neck MRI with 3D inputs.

Sampling is seeded, so the same ``--seed`` over the same source data
reproduces the same manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RealEnv.cases import (BRATS_COHORTS, CaseManifest, EXPECTED_COUNTS,  # noqa: E402
                           ManifestError, RealCase)

GROUNDING_QUESTION = (
    "Locate the {target} in this chest radiograph and describe any abnormality "
    "present in that region."
)
DIAGNOSIS_QUESTION = (
    "Based on this brain MRI and the available grounding evidence, what is the "
    "most likely diagnosis?"
)
REPORT_QUESTION = (
    "Generate a radiology report for this brain MRI, covering the findings and "
    "an impression."
)


# --------------------------------------------------------------------------- #
# Chest Imagenome
# --------------------------------------------------------------------------- #

def collect_chest_imagenome(root: Path, limit: int, rng: random.Random) -> List[RealCase]:
    """Scene graphs to grounding cases.

    Expects the standard release layout, with per-study scene-graph JSON under
    silver_dataset/scene_graph/ and images reachable by study id.
    """
    scene_dir = _first_existing(root, [
        "silver_dataset/scene_graph", "scene_graph", "gold_dataset/scene_graph",
    ])
    if scene_dir is None:
        raise FileNotFoundError(
            f"No scene_graph directory under {root}. Expected one of "
            f"silver_dataset/scene_graph, scene_graph, gold_dataset/scene_graph."
        )

    eligible: List[RealCase] = []
    for path in sorted(scene_dir.rglob("*.json")):
        try:
            graph = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        objects = graph.get("objects") or []
        attributes = graph.get("attributes") or []
        # Criterion: non-empty *local anatomical* and *abnormality* annotations.
        regions = [o for o in objects if o.get("bbox_name")]
        findings = [
            a for a in attributes
            if any("yes|" in str(c) for c in (a.get("attributes") or []))
        ]
        if not regions or not findings:
            continue

        target = findings[0].get("bbox_name") or regions[0]["bbox_name"]
        if not target or target.lower() in {"unknown", "other"}:
            continue  # ambiguous target

        study_id = graph.get("image_id") or path.stem
        eligible.append(RealCase(
            case_id=f"CIG-{study_id}",
            task="organ_anomaly_grounding",
            source="Chest Imagenome",
            anatomy="Chest",
            modality="X-ray",
            dimensionality="2D",
            question=GROUNDING_QUESTION.format(target=target),
            # The target region is stated in the question, so it is an input.
            inputs={"$OrganObject$": target},
            reference={
                "target_region": target,
                "regions": [o["bbox_name"] for o in regions],
                "findings": _finding_labels(findings),
                "scene_graph": str(path.relative_to(root)),
            },
            notes="Reference derived from localized anatomical and abnormality annotations.",
        ))

    if len(eligible) < limit:
        raise ManifestError(
            f"Only {len(eligible)} eligible Chest Imagenome cases found, need "
            f"{limit}. Check that scene graphs and images are both present."
        )

    # Diverse anatomical targets: sample round-robin over target regions
    # before falling back to random choice within a region.
    return _diverse_sample(eligible, limit, key=lambda c: c.reference["target_region"], rng=rng)


def _finding_labels(findings: List[Dict]) -> List[str]:
    labels = []
    for finding in findings:
        for attribute in finding.get("attributes") or []:
            for entry in (attribute if isinstance(attribute, list) else [attribute]):
                if "yes|" in str(entry):
                    labels.append(str(entry).split("|")[-1])
    return sorted(set(labels))


# --------------------------------------------------------------------------- #
# RadGenome-BrainMRI
# --------------------------------------------------------------------------- #

def collect_radgenome(root: Path, rng: random.Random) -> List[RealCase]:
    """BraTS cohorts to 55 diagnosis + 55 report-generation cases."""
    by_cohort: Dict[str, List[Dict]] = {}
    for cohort in BRATS_COHORTS:
        records = _load_cohort(root, cohort)
        if records:
            by_cohort[cohort] = records

    if not by_cohort:
        raise FileNotFoundError(
            f"No BraTS cohort data under {root}. Expected subdirectories or "
            f"report files named for {', '.join(BRATS_COHORTS)}."
        )

    needed = EXPECTED_COUNTS["diagnosis_with_clues"] + EXPECTED_COUNTS["report_generation"]
    balanced = _balanced_sample(by_cohort, needed, rng)
    if len(balanced) < needed:
        raise ManifestError(
            f"Only {len(balanced)} usable RadGenome-BrainMRI cases found, need "
            f"{needed} (55 diagnosis + 55 report generation)."
        )

    cases: List[RealCase] = []
    split = EXPECTED_COUNTS["diagnosis_with_clues"]
    for index, (cohort, record) in enumerate(balanced):
        is_diagnosis = index < split
        case_id = record.get("case_id") or record.get("image_path", f"{cohort}-{index}")
        cases.append(RealCase(
            case_id=f"RGB-{Path(str(case_id)).stem}",
            task="diagnosis_with_clues" if is_diagnosis else "report_generation",
            source="RadGenome-BrainMRI",
            anatomy="Head & Neck",
            modality="MRI",
            dimensionality="3D",
            question=DIAGNOSIS_QUESTION if is_diagnosis else REPORT_QUESTION,
            cohort=cohort,
            reference=(
                {"diagnosis": _diagnosis_for(cohort, record),
                 "lesion_annotation": record.get("mask_path", "")}
                if is_diagnosis else
                {"finding": record.get("finding", ""),
                 "impression": record.get("impression", ""),
                 "lesion_annotation": record.get("mask_path", "")}
            ),
        ))
    return cases


COHORT_DIAGNOSIS = {
    "BraTS-GLI": "glioma",
    "BraTS-MEN": "meningioma",
    "BraTS-MET": "brain metastasis",
}


def _diagnosis_for(cohort: str, record: Dict) -> str:
    return record.get("diagnosis") or COHORT_DIAGNOSIS.get(cohort, "")


def _load_cohort(root: Path, cohort: str) -> List[Dict]:
    """Read a cohort's report/annotation records, tolerating layout variants."""
    records: List[Dict] = []
    for path in sorted(root.rglob("*.json")):
        if cohort.lower().replace("-", "") not in str(path).lower().replace("-", ""):
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        entries = payload if isinstance(payload, list) else payload.get("cases", [])
        for entry in entries:
            if isinstance(entry, dict) and (entry.get("finding") or entry.get("image_path")):
                records.append(entry)
    return records


# --------------------------------------------------------------------------- #
# sampling helpers
# --------------------------------------------------------------------------- #

def _diverse_sample(items, limit, key, rng):
    """Round-robin over distinct key values so one region cannot dominate."""
    buckets: Dict[str, List] = {}
    for item in items:
        buckets.setdefault(key(item), []).append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    chosen, order = [], sorted(buckets)
    while len(chosen) < limit and any(buckets[k] for k in order):
        for k in order:
            if buckets[k] and len(chosen) < limit:
                chosen.append(buckets[k].pop())
    return chosen


def _balanced_sample(by_cohort: Dict[str, List[Dict]], needed: int, rng):
    """Round-robin across cohorts, as balanced as availability allows."""
    pools = {c: list(records) for c, records in by_cohort.items()}
    for pool in pools.values():
        rng.shuffle(pool)

    chosen, order = [], sorted(pools)
    while len(chosen) < needed and any(pools[c] for c in order):
        for cohort in order:
            if pools[cohort] and len(chosen) < needed:
                chosen.append((cohort, pools[cohort].pop()))
    return chosen


def _first_existing(root: Path, candidates: List[str]):
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    return None


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chest-imagenome", required=True, type=Path,
                        help="root of your Chest Imagenome release")
    parser.add_argument("--radgenome", required=True, type=Path,
                        help="root of your RadGenome-BrainMRI copy")
    parser.add_argument("--out", default="RealEnv/manifest.json", type=Path)
    parser.add_argument("--seed", type=int, default=2024,
                        help="sampling seed; the same seed reproduces the same manifest")
    args = parser.parse_args()

    for label, path in (("--chest-imagenome", args.chest_imagenome),
                        ("--radgenome", args.radgenome)):
        if not path.is_dir():
            parser.error(f"{label} is not a directory: {path}")

    rng = random.Random(args.seed)

    print("Collecting Chest Imagenome grounding cases...")
    grounding = collect_chest_imagenome(
        args.chest_imagenome, EXPECTED_COUNTS["organ_anomaly_grounding"], rng)
    print(f"  {len(grounding)} cases")

    print("Collecting RadGenome-BrainMRI cases...")
    brain = collect_radgenome(args.radgenome, rng)
    print(f"  {len(brain)} cases")

    manifest = CaseManifest(grounding + brain)
    manifest.validate()
    manifest.save(str(args.out), metadata={
        "seed": args.seed,
        "chest_imagenome_root": str(args.chest_imagenome),
        "radgenome_root": str(args.radgenome),
    })

    print()
    print(manifest.resolution_report())
    print(f"\nManifest written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
