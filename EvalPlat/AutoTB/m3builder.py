"""Real-environment AutoTB: drive M3Builder and validate what it produces.

M3Builder (https://github.com/MAGIC-AI4Med/M3Builder) is a four-agent system
entered through:

    python3 AgentPipelines/run.py --human_requirements "<free text>"

Its agents run in sequence (Dataset Selector, Data Processor, Dataloader
Builder, Model Trainer) and write logs and checkpoints under
TrainPipeline/Logout. This wrapper attempts a build, validates it against a
ToolCard, and either registers it in the executable pool or reports a
construction failure.

Each stage is recorded separately so failures can be attributed. Anything that
fails before a validated candidate exists (dataset selection, a non-executable
dataloader, training termination, a missing checkpoint, no evaluation output)
is a construction failure and is not registered. A downstream call with an
incompatible input against a registered tool is a usage failure.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .tool_card import ToolCard

__all__ = ["BuildOutcome", "M3BuilderRunner", "ToolPool"]

# Stages the Model Trainer prints, in order. Where a run stops tells us where
# construction broke.
TRAINER_STAGES = ("Setup Data", "Setup Model", "Setup Trainer",
                  "Start Training", "trainer.evaluate")


@dataclass
class BuildOutcome:
    """Everything needed to attribute a success or failure."""

    requested: Dict
    status: str = "NOT_RUN"          # SUCCESS | CONSTRUCTION_FAILURE | ENV_FAILURE
    stage_reached: str = ""
    validation: str = "NOT_RUN"      # PASSED | FAILED | NOT_RUN
    artifacts: List[str] = field(default_factory=list)
    build_seconds: float = 0.0
    trials: int = 0
    log_path: str = ""
    message: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


class ToolPool:
    """The executable tool registry, persisted as JSON.

    Tools built by AutoTB are appended here and become available on replanning.
    Registration is a separate step from building: only a validated candidate
    is registered.
    """

    def __init__(self, path: str = "RealEnv/tool_pool.json") -> None:
        self.path = Path(path)
        self.tools: List[Dict] = []
        if self.path.is_file():
            self.tools = json.loads(self.path.read_text())

    def register(self, card: ToolCard, checkpoint: str, outcome: BuildOutcome) -> str:
        tool_id = f"AutoTB-{_abbrev(card.category)}-{len(self.tools):03d}"
        self.tools.append({
            "tool_id": tool_id,
            **card.as_dict(),
            "checkpoint": checkpoint,
            "registry_status": "ACTIVE",
            "build_status": outcome.status,
            "build_seconds": outcome.build_seconds,
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.tools, indent=2))
        return tool_id

    def covers(self, card: ToolCard) -> bool:
        """True when the pool already holds a tool matching this card."""
        return any(
            t["category"] == card.category
            and t["anatomy"] in (card.anatomy, "Universal")
            and t["modality"] in (card.modality, "Universal")
            for t in self.tools
        )


class M3BuilderRunner:
    """Invoke M3Builder for one tool card and validate the result."""

    def __init__(self, m3builder_root: str, max_trials: int = 3, timeout: int = 7200) -> None:
        self.root = Path(m3builder_root)
        self.max_trials = max_trials
        self.timeout = timeout
        entry = self.root / "AgentPipelines" / "run.py"
        if not entry.is_file():
            raise FileNotFoundError(
                f"M3Builder not found at {self.root}. Clone it first:\n"
                f"    git clone https://github.com/MAGIC-AI4Med/M3Builder.git\n"
                f"then pass its path with --m3builder-root."
            )
        self.entry = entry

    # -- Algorithm 1, lines 7-11 -------------------------------------------

    def build(self, card: ToolCard, dimensionality: str = "2D") -> BuildOutcome:
        """Attempt the build, retrying up to max_trials times.

        M3Builder self-corrects across rounds, so each trial sees the previous
        failure, matching M3Builder(C_{k-1}, R_{k-1}, g | P) in the pseudo-code.
        """
        outcome = BuildOutcome(requested=card.as_dict())
        requirements = card.human_requirements(dimensionality)

        for trial in range(1, self.max_trials + 1):
            outcome.trials = trial
            started = time.time()
            try:
                proc = subprocess.run(
                    ["python3", str(self.entry), "--human_requirements", requirements],
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                outcome.status = "CONSTRUCTION_FAILURE"
                outcome.stage_reached = "TIMEOUT"
                outcome.message = f"M3Builder exceeded {self.timeout}s on trial {trial}."
                continue
            except OSError as error:
                outcome.status = "ENV_FAILURE"
                outcome.message = str(error)
                return outcome

            outcome.build_seconds += time.time() - started
            log = proc.stdout + proc.stderr
            outcome.stage_reached = _last_stage(log)
            outcome.log_path = self._dump_log(card, trial, log)

            checkpoint = _find_checkpoint(self.root)
            if proc.returncode == 0 and checkpoint:
                outcome.artifacts = [outcome.log_path, checkpoint]
                outcome.status = "SUCCESS"
                return outcome

            outcome.status = "CONSTRUCTION_FAILURE"
            outcome.message = (
                f"trial {trial}: exit={proc.returncode}, "
                f"last stage={outcome.stage_reached or 'none'}, "
                f"checkpoint={'found' if checkpoint else 'missing'}"
            )

        return outcome

    # -- Algorithm 1, line 9 ------------------------------------------------

    def validate(self, outcome: BuildOutcome, card: ToolCard) -> bool:
        """Check the built model against the card before registering it.

        A model that trains but does not provide the requested capability must
        not enter the pool, or a later failure is misattributed to usage.
        """
        if outcome.status != "SUCCESS":
            outcome.validation = "NOT_RUN"
            return False

        log = Path(outcome.log_path).read_text() if outcome.log_path else ""
        reached_eval = "trainer.evaluate" in log or "evaluation" in log.lower()
        has_checkpoint = any("checkpoint" in a.lower() for a in outcome.artifacts)
        task_matches = card.m3_task.lower().replace("_", "") in log.lower().replace("_", "")

        passed = reached_eval and has_checkpoint
        outcome.validation = "PASSED" if passed else "FAILED"
        if not passed:
            outcome.status = "CONSTRUCTION_FAILURE"
            outcome.message = (
                "Training finished but validation failed: "
                f"evaluation={'yes' if reached_eval else 'no'}, "
                f"checkpoint={'yes' if has_checkpoint else 'no'}, "
                f"task match={'yes' if task_matches else 'no'}."
            )
        return passed

    def _dump_log(self, card: ToolCard, trial: int, log: str) -> str:
        out_dir = self.root / "TrainPipeline" / "Logout" / "AutoTB"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{_abbrev(card.category)}_trial{trial}.log"
        path.write_text(log)
        return str(path)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _abbrev(category: str) -> str:
    return "".join(word[0] for word in category.split() if word).upper() or "TOOL"


def _last_stage(log: str) -> str:
    reached = [stage for stage in TRAINER_STAGES if stage in log]
    return reached[-1] if reached else ""


def _find_checkpoint(root: Path) -> Optional[str]:
    logout = root / "TrainPipeline" / "Logout"
    if not logout.is_dir():
        return None
    candidates = sorted(
        (p for p in logout.rglob("*") if re.search(r"\.(pt|pth|bin|safetensors)$", p.name)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None
