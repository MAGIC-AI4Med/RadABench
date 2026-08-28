"""The 165 real-environment cases: schema, loading and validation.

These cases are not converted into patient records. Each keeps its original
clinical question, imaging data and reference annotation or report, so the
agent works under real data constraints (sec:real_env_data).

    organ_anomaly_grounding   55   Chest Imagenome      2D
    diagnosis_with_clues      55   RadGenome-BrainMRI   3D
    report_generation         55   RadGenome-BrainMRI   3D

The manifest ships without image paths. Chest Imagenome derives from MIMIC-CXR
and RadGenome-BrainMRI from the BraTS cohorts, both credentialed, so the pixels
cannot be redistributed here. What ships is the case id list, the task
assignment and the reference signal, which is enough to rebuild the exact
evaluation set from your own copies via build_manifest.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

__all__ = ["RealCase", "CaseManifest", "TASKS", "EXPECTED_COUNTS", "ManifestError"]

TASKS = ("organ_anomaly_grounding", "diagnosis_with_clues", "report_generation")

# Table tab:real_environment_case_sources. A manifest that disagrees with these
# counts is an error, not a warning: a 54-case run reported as the 55-case
# benchmark is the drift this guards against.
EXPECTED_COUNTS = {
    "organ_anomaly_grounding": 55,
    "diagnosis_with_clues": 55,
    "report_generation": 55,
}

SOURCE_OF = {
    "organ_anomaly_grounding": "Chest Imagenome",
    "diagnosis_with_clues": "RadGenome-BrainMRI",
    "report_generation": "RadGenome-BrainMRI",
}

# RadGenome-BrainMRI draws from three multi-institutional BraTS cohorts.
# Sampling is balanced across them as far as availability allows.
BRATS_COHORTS = ("BraTS-GLI", "BraTS-MEN", "BraTS-MET")


class ManifestError(ValueError):
    """Raised when a manifest does not describe the benchmark it claims to."""


@dataclass
class RealCase:
    """One real case, as the agent and the scorer see it."""

    case_id: str
    task: str
    source: str
    anatomy: str
    modality: str
    dimensionality: str
    question: str
    # Resolved locally from --data-root; empty in the shipped manifest.
    image_path: str = ""
    # Variables the clinical question already supplies, e.g. which organ to
    # segment. These seed the agent's memory bank. Keep separate from
    # `reference`: the target region is part of the question, the mask is the
    # answer, and merging them would leak the label into the input.
    inputs: Dict[str, str] = field(default_factory=dict)
    # The reference signal: a mask path for grounding, a label for diagnosis,
    # report text for report generation.
    reference: Dict = field(default_factory=dict)
    cohort: str = ""
    notes: str = ""

    def is_resolved(self) -> bool:
        """True when the local image file has been located."""
        return bool(self.image_path) and Path(self.image_path).exists()

    def as_dict(self) -> Dict:
        return asdict(self)


class CaseManifest:
    """The evaluation set, loaded and checked as a unit."""

    def __init__(self, cases: Iterable[RealCase]) -> None:
        self.cases: List[RealCase] = list(cases)

    # -- io -----------------------------------------------------------------

    @classmethod
    def load(cls, path: str, validate: bool = True) -> "CaseManifest":
        payload = json.loads(Path(path).read_text())
        manifest = cls(RealCase(**record) for record in payload["cases"])
        if validate:
            manifest.validate()
        return manifest

    def save(self, path: str, metadata: Optional[Dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "metadata": {
                "benchmark": "RadA-BenchPlat real-environment layer",
                "total_cases": len(self.cases),
                "counts": self.counts(),
                **(metadata or {}),
            },
            "cases": [c.as_dict() for c in self.cases],
        }, indent=2))

    # -- querying -----------------------------------------------------------

    def counts(self) -> Dict[str, int]:
        return {task: sum(1 for c in self.cases if c.task == task) for task in TASKS}

    def by_task(self, task: str) -> List[RealCase]:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
        return [c for c in self.cases if c.task == task]

    def resolved(self) -> List[RealCase]:
        return [c for c in self.cases if c.is_resolved()]

    # -- checking -----------------------------------------------------------

    def validate(self) -> None:
        """Raise when the manifest is not the published evaluation set."""
        problems: List[str] = []

        counts = self.counts()
        for task, expected in EXPECTED_COUNTS.items():
            if counts[task] != expected:
                problems.append(
                    f"{task}: {counts[task]} cases, expected {expected}")

        total = sum(EXPECTED_COUNTS.values())
        if len(self.cases) != total:
            problems.append(f"total: {len(self.cases)} cases, expected {total}")

        ids = [c.case_id for c in self.cases]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            problems.append(f"duplicate case ids: {sorted(duplicates)}")

        for case in self.cases:
            if case.task not in TASKS:
                problems.append(f"{case.case_id}: unknown task {case.task!r}")
            if not case.reference:
                problems.append(f"{case.case_id}: empty reference signal")

        if problems:
            raise ManifestError(
                "Manifest does not match the published evaluation set:\n  - "
                + "\n  - ".join(problems)
            )

    def resolution_report(self) -> str:
        """Human-readable summary of how much of the manifest is runnable."""
        lines = [f"{'Task':<28}{'Cases':>7}{'Resolved':>10}{'Source':>22}"]
        lines.append("-" * 67)
        for task in TASKS:
            cases = self.by_task(task)
            resolved = sum(1 for c in cases if c.is_resolved())
            lines.append(f"{task:<28}{len(cases):>7}{resolved:>10}{SOURCE_OF[task]:>22}")
        total_resolved = len(self.resolved())
        lines.append("-" * 67)
        lines.append(f"{'Total':<28}{len(self.cases):>7}{total_resolved:>10}")
        if total_resolved < len(self.cases):
            lines.append("")
            lines.append("Unresolved cases have no local image file. Run "
                         "build_manifest.py with --data-root pointing at your "
                         "copies of Chest Imagenome and RadGenome-BrainMRI.")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)
