"""Executable tool pool for the real-environment layer.

Fourteen dataset-specific tool families cover four executable task groups:
organ segmentation, anomaly detection, disease diagnosis, report generation.
M3Builder trains each family three times on different data-subset
configurations, giving low / medium / high performing variants of the same
declared capability, for 42 concrete tools (Table tab:m3builder_tool_pool).

Three variants per family is what allows the perturbation settings to run on
real tools. Differentiated needs several genuinely suitable candidates that
differ in quality; Redundant needs tools that look alike in name but differ in
anatomy, modality, dimensionality or I/O schema.

The agent sees only the tool cards, never the underlying accuracy, and must
infer usability from the declared capability constraints.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

__all__ = ["ExecutableTool", "ToolFamily", "TOOL_FAMILIES", "ExecutableToolPool",
           "build_default_pool"]

# Variant suffixes, in ascending order of training data / rounds.
VARIANT_TIERS = ("v1", "v2", "v3")


@dataclass(frozen=True)
class ToolFamily:
    """One dataset-specific capability, before variant expansion."""

    name: str
    category: str
    dataset: str
    anatomy: str
    modality: str
    dimensionality: str
    capability: str
    compulsory_input: List[str]
    output: List[str]
    # Performance of the three variants, weakest first. Placeholders until a
    # validation round fills them in; see calibrate_from_results().
    variant_scores: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


# Table `tab:m3builder_tool_pool`, row for row.
TOOL_FAMILIES: List[ToolFamily] = [
    # -- Organ Segmentor ----------------------------------------------------
    ToolFamily("OS-BTCV", "Organ Segmentor", "BTCV", "Abdomen & Pelvis", "CT", "3D",
               "Multi-organ abdominal and pelvic CT segmentation.",
               ["$Image$", "$OrganObject$"], ["$OrganDim$"]),
    ToolFamily("OS-VerSe", "Organ Segmentor", "VerSe", "Spine", "CT", "3D",
               "Vertebra/spine structure segmentation on CT.",
               ["$Image$", "$OrganObject$"], ["$OrganDim$"]),
    ToolFamily("OS-OASIS", "Organ Segmentor", "OASIS", "Head & Neck", "MRI", "3D",
               "Brain/head-and-neck structure segmentation on MRI.",
               ["$Image$", "$OrganObject$"], ["$OrganDim$"]),
    # -- Anomaly Detector ---------------------------------------------------
    ToolFamily("AD-COV19", "Anomaly Detector", "COVID19", "Chest", "CT", "3D",
               "Chest CT abnormality detection and localization.",
               ["$Image$"], ["$AnomalyObject$", "$AnomalyDim$"]),
    ToolFamily("AD-INS22", "Anomaly Detector", "Instance22", "Head & Neck", "CT", "3D",
               "Head-and-neck CT anomaly localization.",
               ["$Image$"], ["$AnomalyObject$", "$AnomalyDim$"]),
    ToolFamily("AD-Panc", "Anomaly Detector", "MSD_Pancreas", "Abdomen & Pelvis", "CT", "3D",
               "Pancreatic-region anomaly detection on CT.",
               ["$Image$"], ["$AnomalyObject$", "$AnomalyDim$"]),
    ToolFamily("AD-XDet10", "Anomaly Detector", "ChestX-Det10", "Chest", "X-ray", "2D",
               "Chest X-ray anomaly detection and localization.",
               ["$Image$"], ["$AnomalyObject$", "$AnomalyDim$"]),
    # -- Imaging Diagnoser (code name: Disease Diagnoser) -------------------
    ToolFamily("ID-ADNI", "Disease Diagnoser", "ADNI", "Head & Neck", "MRI", "3D",
               "MRI-based neuroimaging disease diagnosis.",
               ["$Image$"], ["$Disease$"]),
    ToolFamily("ID-KneeMR", "Disease Diagnoser", "KneeMRI", "Limb", "MRI", "2D",
               "Knee MRI disease or abnormality diagnosis.",
               ["$Image$"], ["$Disease$"]),
    ToolFamily("ID-CC-CCII", "Disease Diagnoser", "CC-CCII", "Chest", "CT", "3D",
               "Chest CT disease diagnosis.",
               ["$Image$"], ["$Disease$"]),
    ToolFamily("ID-KidCT", "Disease Diagnoser", "CT-kidney", "Abdomen & Pelvis", "CT", "2D",
               "Kidney/abdominal CT disease diagnosis.",
               ["$Image$"], ["$Disease$"]),
    # -- Report Generator ---------------------------------------------------
    ToolFamily("RG-CT-RATE", "Report Generator", "CT-RATE", "Chest", "CT", "3D",
               "Chest CT report generation.",
               ["$Image$"], ["$Report$"]),
    ToolFamily("RG-GenomBra", "Report Generator", "BrainGnome", "Head & Neck", "MRI", "3D",
               "Brain MRI report generation.",
               ["$Image$"], ["$Report$"]),
    ToolFamily("RG-IU-Xray", "Report Generator", "IU-Xray", "Chest", "X-ray", "2D",
               "Chest X-ray report generation.",
               ["$Image$"], ["$Report$"]),
]

# Ready-made tools, not trained by M3Builder, used as alternatives for
# segmentation and diagnosis (paper 2.1).
EXTERNAL_TOOLS = [
    ToolFamily("OS-nnUNet", "Organ Segmentor", "nnU-Net", "Universal", "Universal", "3D",
               "Self-configuring segmentation across anatomies and modalities.",
               ["$Image$", "$OrganObject$"], ["$OrganDim$"], [0.0]),
    ToolFamily("ID-RP3D-DM", "Disease Diagnoser", "RP3D-DM", "Universal", "Universal", "3D",
               "Large-scale radiology diagnosis across anatomies.",
               ["$Image$"], ["$Disease$"], [0.0]),
]


@dataclass
class ExecutableTool:
    """One concrete, callable tool variant.

    performance is benchmark metadata driving the Differentiated setting's
    ranking. The agent is shown the card without it unless the condition
    reveals it.
    """

    tool_id: str
    category: str
    family: str
    dataset: str
    anatomy: str
    modality: str
    dimensionality: str
    capability: str
    compulsory_input: List[str]
    optional_input: List[str] = field(default_factory=list)
    output: List[str] = field(default_factory=list)
    unsupported: Dict[str, List[str]] = field(default_factory=dict)
    performance: float = 0.0
    variant: str = "v1"
    checkpoint: str = ""
    source: str = "M3Builder"
    registry_status: str = "ACTIVE"

    def supports(self, anatomy: str, modality: str, dimensionality: str = "") -> bool:
        """Can this tool legitimately be called on such a case?

        Universal scope matches anything, which is why nnU-Net and RP3D-DM are
        usable everywhere and why requesting a universal tool under config2
        counts as grounded.
        """
        def matches(declared: str, wanted: str) -> bool:
            return not wanted or declared == "Universal" or declared == wanted

        return (matches(self.anatomy, anatomy)
                and matches(self.modality, modality)
                and matches(self.dimensionality, dimensionality))

    def card(self, reveal_performance: bool = False) -> Dict:
        """The structured description shown to the agent."""
        data = {
            "tool_id": self.tool_id,
            "category": self.category,
            "property": self.capability,
            "anatomy": self.anatomy,
            "modality": self.modality,
            "dimensionality": self.dimensionality,
            "compulsory_input": self.compulsory_input,
            "optional_input": self.optional_input,
            "output": self.output,
            "capability_constraints": self.unsupported,
        }
        if reveal_performance:
            data["performance"] = self.performance
        return data

    def as_dict(self) -> Dict:
        return asdict(self)


class ExecutableToolPool:
    """Registry of executable tools, including any AutoTB adds at run time."""

    def __init__(self, tools: Optional[Iterable[ExecutableTool]] = None) -> None:
        self.tools: List[ExecutableTool] = list(tools or [])

    # -- construction -------------------------------------------------------

    @classmethod
    def from_json(cls, path: str) -> "ExecutableToolPool":
        records = json.loads(Path(path).read_text())
        return cls(ExecutableTool(**r) for r in records)

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([t.as_dict() for t in self.tools], indent=2))

    # -- querying -----------------------------------------------------------

    def by_id(self, tool_id: str) -> Optional[ExecutableTool]:
        return next((t for t in self.tools if t.tool_id == tool_id), None)

    def candidates(self, category: str, anatomy: str = "", modality: str = "",
                   dimensionality: str = "") -> List[ExecutableTool]:
        """Every tool that could serve this step, best first.

        Sorting by performance is what lets the Differentiated setting score
        OTS: the rank of the agent's pick in this list is R_t.
        """
        matching = [
            t for t in self.tools
            if t.category == category
            and t.registry_status == "ACTIVE"
            and t.supports(anatomy, modality, dimensionality)
        ]
        return sorted(matching, key=lambda t: t.performance, reverse=True)

    def rank_of(self, tool_id: str, category: str, anatomy: str = "",
                modality: str = "", dimensionality: str = "") -> Optional[int]:
        """1-based performance rank of a chosen tool among its alternatives."""
        options = self.candidates(category, anatomy, modality, dimensionality)
        for index, tool in enumerate(options, start=1):
            if tool.tool_id == tool_id:
                return index
        return None

    # -- perturbation -------------------------------------------------------

    def without_category(self, category: str) -> "ExecutableToolPool":
        """Insufficient config1: remove an entire category."""
        return ExecutableToolPool(t for t in self.tools if t.category != category)

    def without_capability(self, category: str, anatomy: str, modality: str) -> "ExecutableToolPool":
        """Insufficient config2: keep the category, remove the coverage."""
        return ExecutableToolPool(
            t for t in self.tools
            if not (t.category == category and t.supports(anatomy, modality))
        )

    def best_only(self) -> "ExecutableToolPool":
        """Keep only the strongest variant per family (the Baseline setting)."""
        best: Dict[str, ExecutableTool] = {}
        for tool in self.tools:
            current = best.get(tool.family)
            if current is None or tool.performance > current.performance:
                best[tool.family] = tool
        return ExecutableToolPool(best.values())

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self):
        return iter(self.tools)


def build_default_pool(include_external: bool = True) -> ExecutableToolPool:
    """Expand the 14 families into their 42 variants, plus external tools.

    Performance values start at 0.0, ordered only by variant tier. Fill them
    from real validation runs with calibrate_from_results before using the
    Differentiated setting, or every tool ties and OTS is meaningless.
    """
    tools: List[ExecutableTool] = []
    for family in TOOL_FAMILIES:
        for tier_index, tier in enumerate(VARIANT_TIERS):
            tools.append(ExecutableTool(
                tool_id=f"{family.name}-{tier}",
                category=family.category,
                family=family.name,
                dataset=family.dataset,
                anatomy=family.anatomy,
                modality=family.modality,
                dimensionality=family.dimensionality,
                capability=f"{family.capability} (subset-trained variant {tier})",
                compulsory_input=list(family.compulsory_input),
                output=list(family.output),
                performance=family.variant_scores[tier_index]
                if tier_index < len(family.variant_scores) else 0.0,
                variant=tier,
            ))

    if include_external:
        for family in EXTERNAL_TOOLS:
            tools.append(ExecutableTool(
                tool_id=family.name,
                category=family.category,
                family=family.name,
                dataset=family.dataset,
                anatomy=family.anatomy,
                modality=family.modality,
                dimensionality=family.dimensionality,
                capability=family.capability,
                compulsory_input=list(family.compulsory_input),
                output=list(family.output),
                source=family.dataset,
            ))

    return ExecutableToolPool(tools)


def calibrate_from_results(pool: ExecutableToolPool, scores: Dict[str, float]) -> ExecutableToolPool:
    """Attach measured validation performance to tools, keyed by tool_id.

    Tools absent from scores keep 0.0 and sort last, so an uncalibrated tool is
    never presented as the best option.
    """
    for tool in pool:
        if tool.tool_id in scores:
            tool.performance = scores[tool.tool_id]
    return pool
