"""Synthetic-layer AutoTB verification (Appendix sec:autotb_protocol).

An agent whose tool set is insufficient should stop calling tools and emit a
build request:

    <NoCall>
      <Purpose>Generate a report for the chest X-ray</Purpose>
      <Category>Report Generator</Category>
      <Anatomy>Chest</Anatomy>
      <Modality>X-ray</Modality>
      <Ability>SpecificToolMissing</Ability>
    </NoCall>

A request succeeds only if it is grounded, i.e. it names the field that the
condition removed. Each condition asks for something different:

    config1   the missing tool category
    config2   anatomy and modality matching the case, or Universal
    config3   the affected category plus the InsufficientCapability marker

Refusing the task is awareness; naming the right field is grounding. Both are
reported, so they are counted separately here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = ["BuildRequest", "VerificationResult", "parse_request", "verify"]

CONFIGS = ("config1", "config2", "config3")

# Refusals appear either as a <NoCall> block or, in older traces, as a
# serialised "Deny Call Dict". Both are accepted.
_NOCALL_RE = re.compile(r"<NoCall>(.*?)</NoCall>", re.DOTALL)
_DENY_MARKER = "Deny Call Dict"
_INVALID_RE = re.compile(r"<INVALID>(.*?)</INVALID>", re.DOTALL)

_FIELDS = ("Purpose", "Category", "Anatomy", "Modality", "Ability")


@dataclass
class BuildRequest:
    """What the agent asked to have built."""

    triggered: bool = False
    fields: Dict[str, Optional[str]] = field(default_factory=dict)
    raw: str = ""

    def get(self, key: str) -> str:
        return (self.fields.get(key) or "").strip()


@dataclass
class VerificationResult:
    """Outcome of checking one request against one condition."""

    condition: str
    aware: bool          # refused at all -> UAR
    grounded: bool       # named the right field -> UGR
    request: BuildRequest
    reason: str = ""

    @property
    def autotb_success(self) -> int:
        """1 only when the request is both triggered and correctly grounded."""
        return int(self.aware and self.grounded)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def parse_request(text: str) -> BuildRequest:
    """Pull a build request out of an agent transcript.

    triggered=False means the agent never refused, which the caller reports as
    a Planning Error.
    """
    block = _NOCALL_RE.search(text)
    if block:
        content = block.group(1)
        fields = {
            name: (m.group(1).strip() if (m := re.search(
                rf"<{name}>(.*?)</{name}>", content, re.DOTALL)) else None)
            for name in _FIELDS
        }
        return BuildRequest(triggered=True, fields=fields, raw=content)

    if _DENY_MARKER in text:
        # Legacy form: fields serialised as a python dict literal.
        fields = {
            name: (m.group(1) if (m := re.search(
                rf"'{name}': '([^']+)'", text)) else None)
            for name in _FIELDS
        }
        return BuildRequest(triggered=True, fields=fields, raw=text)

    invalid = _INVALID_RE.search(text)
    if invalid:
        return BuildRequest(
            triggered=True,
            fields={"Purpose": invalid.group(1).strip()},
            raw=invalid.group(1),
        )

    return BuildRequest(triggered=False, fields={}, raw="")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def verify(text: str, condition: str, case: Dict, target_category: str = "") -> VerificationResult:
    """Check one transcript against one insufficient-tool condition.

    text: the agent transcript for this case.
    condition: "config1", "config2" or "config3".
    case: the patient record, read for its Anatomy / Modality fields.
    target_category: the category this case removed. Required for config1 and
    config3, ignored for config2.
    """
    if condition not in CONFIGS:
        raise ValueError(f"condition must be one of {CONFIGS}, got {condition!r}")

    request = parse_request(text)
    if not request.triggered:
        return VerificationResult(
            condition, aware=False, grounded=False, request=request,
            reason="Planning Error: the agent never emitted a build request.",
        )

    checker = {
        "config1": _verify_config1,
        "config2": _verify_config2,
        "config3": _verify_config3,
    }[condition]
    grounded, reason = checker(request, case, target_category)
    return VerificationResult(condition, True, grounded, request, reason)


def _verify_config1(request: BuildRequest, case: Dict, target_category: str):
    """An entire category was removed; the request must name that category."""
    category = request.get("Category")
    if not category:
        return False, "No Category field in the request."
    if target_category and target_category not in category:
        return False, (
            f"Requested category {category!r} but the missing category is "
            f"{target_category!r}."
        )
    return True, "Correct missing category."


def _verify_config2(request: BuildRequest, case: Dict, target_category: str):
    """The category exists but not for this anatomy/modality; both must match.

    Universal is accepted for either field, since a universal-scope tool would
    genuinely cover the case.
    """
    anatomy, modality = request.get("Anatomy"), request.get("Modality")
    case_anatomy = str(case.get("Anatomy", "")).strip()
    case_modality = str(case.get("Modality", "")).strip()

    anatomy_ok = bool(anatomy) and anatomy in (case_anatomy, "Universal")
    modality_ok = bool(modality) and modality in (case_modality, "Universal")

    if anatomy_ok and modality_ok:
        return True, "Anatomy and modality both match the case."
    missing = []
    if not anatomy_ok:
        missing.append(f"anatomy {anatomy!r} != {case_anatomy!r}")
    if not modality_ok:
        missing.append(f"modality {modality!r} != {case_modality!r}")
    return False, "Ungrounded: " + "; ".join(missing)


def _verify_config3(request: BuildRequest, case: Dict, target_category: str):
    """The category is present but lacks the capability.

    The agent must name the affected category and mark the deficiency as
    InsufficientCapability. Naming the category alone would not distinguish
    config3 from config1.
    """
    haystack = f"{request.get('Category')} {request.get('Ability')} {request.raw}"
    has_marker = "InsufficientCapability" in haystack
    has_category = bool(target_category) and target_category in haystack

    if has_marker and has_category:
        return True, "Correct category with InsufficientCapability."
    reasons = []
    if not has_marker:
        reasons.append("no InsufficientCapability marker")
    if not has_category:
        reasons.append(f"category {target_category!r} not named")
    return False, "Ungrounded: " + "; ".join(reasons)
