"""Turn a verified build request into a tool card (Algorithm 1, line 5).

In the synthetic layer the card is the deliverable: nothing is trained, and the
card becomes available so the rest of the chain can be evaluated. In the real
layer the same card is the specification M3Builder must satisfy and the
contract the built model is validated against.

One card format for both layers keeps construction failures separable from
usage errors, since the card fixes the I/O schema up front.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

__all__ = ["ToolCard", "specify_tool_card", "CATEGORY_IO"]

# Input/output schema per tool category, matching Utils/tooluse.py.
CATEGORY_IO: Dict[str, Dict[str, List[str]]] = {
    "Anatomy Classifier":   {"in": ["$Image$"],                        "out": ["$Anatomy$"]},
    "Modality Classifier":  {"in": ["$Image$"],                        "out": ["$Modality$"]},
    "Organ Segmentor":      {"in": ["$Image$", "$OrganObject$"],       "out": ["$OrganDim$"]},
    "Anomaly Detector":     {"in": ["$Image$"],                        "out": ["$AnomalyObject$", "$AnomalyDim$"]},
    "Disease Diagnoser":    {"in": ["$Image$"],                        "out": ["$Disease$"]},
    "Disease Inferencer":   {"in": ["$Image$", "$AnomalyObject$"],     "out": ["$Disease$"]},
    "Biomarker Quantifier": {"in": ["$Image$", "$OrganDim$"],          "out": ["$OrganQuant$"]},
    "Indicator Evaluator":  {"in": ["$Disease$", "$OrganQuant$"],      "out": ["$IndicatorName$", "$IndicatorValue$"]},
    "Report Generator":     {"in": ["$Image$"],                        "out": ["$Report$"]},
    "Treatment Recommender": {"in": ["$Disease$", "$Report$"],         "out": ["$Treatment$"]},
}

# M3Builder trains four task families. A requested category must map onto one
# of them to be executable in the real layer.
CATEGORY_TO_M3_TASK = {
    "Organ Segmentor": "OrganSegmentation",
    "Anomaly Detector": "AnomalySegmentation",
    "Disease Diagnoser": "DiseaseDiagnosis",
    "Disease Inferencer": "DiseaseDiagnosis",
    "Report Generator": "ReportGeneration",
}


@dataclass
class ToolCard:
    """The specification of a tool the agent asked to have built."""

    category: str
    anatomy: str = "Universal"
    modality: str = "Universal"
    compulsory_input: List[str] = field(default_factory=list)
    optional_input: List[str] = field(default_factory=list)
    output: List[str] = field(default_factory=list)
    description: str = ""   # rendered as `property` in trace logs
    performance: float = 0.0
    source: str = "AutoTB"

    # -- real-layer helpers -------------------------------------------------

    @property
    def m3_task(self) -> str:
        """The M3Builder task family this card maps onto."""
        task = CATEGORY_TO_M3_TASK.get(self.category)
        if task is None:
            raise ValueError(
                f"M3Builder cannot train a {self.category!r}. Buildable "
                f"categories: {', '.join(CATEGORY_TO_M3_TASK)}."
            )
        return task

    def human_requirements(self, dimensionality: str = "2D") -> str:
        """The free-text request string M3Builder's run.py takes.

        Its entry point is run.py --human_requirements "<free text>", so the
        card is rendered back into a sentence the Task Manager agent can act on.
        """
        scope = f"{self.anatomy} {self.modality}".replace("Universal", "").strip()
        subject = f"{dimensionality} {scope} images" if scope else f"{dimensionality} images"
        task = {
            "OrganSegmentation": "organ segmentation",
            "AnomalySegmentation": "anomaly segmentation",
            "DiseaseDiagnosis": "disease diagnosis",
            "ReportGeneration": "report generation",
        }[self.m3_task]
        return f"Train a model for {task} using {subject}."

    def as_dict(self) -> Dict:
        return {
            "category": self.category,
            "property": self.description,
            "anatomy": self.anatomy,
            "modality": self.modality,
            "compulsory_input": self.compulsory_input,
            "optional_input": self.optional_input,
            "output": self.output,
            "performance": self.performance,
            "source": self.source,
        }


def specify_tool_card(request, dimensionality: str = "2D") -> ToolCard:
    """Build a card from a verified BuildRequest.

    Only call this on a request that passed verification. An ungrounded request
    would produce a card for the wrong capability.
    """
    category = request.get("Category")
    if not category:
        raise ValueError("Cannot specify a tool card without a Category.")

    # Tolerate close variants, e.g. "Report Generation" for "Report Generator".
    canonical = next(
        (known for known in CATEGORY_IO
         if known.lower() in category.lower() or category.lower() in known.lower()),
        category,
    )
    schema = CATEGORY_IO.get(canonical, {"in": ["$Image$"], "out": []})

    anatomy = request.get("Anatomy") or "Universal"
    modality = request.get("Modality") or "Universal"
    scope = " ".join(p for p in (anatomy, modality) if p != "Universal")

    return ToolCard(
        category=canonical,
        anatomy=anatomy,
        modality=modality,
        compulsory_input=list(schema["in"]),
        output=list(schema["out"]),
        description=f"{scope} {canonical.lower()}".strip(),
    )
