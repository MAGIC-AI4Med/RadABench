# The real-environment layer

165 real cases, executable tools, and no synthetic conversion. Each case keeps
its original clinical question, image, and reference annotation or report, so
the agent has to work under real data constraints rather than idealized ones.

The reason this layer exists: in the synthetic layer a failure belongs to the
agent core, because the tools are perfect by construction. Here a failure can
belong to the agent, the tool, or the interface between them, and telling those
apart is what this layer is for.

## What ships, and what doesn't

| | |
|---|---|
| In this repository | case manifest schema, selection criteria as runnable code, the 42-variant tool pool, all six automatic metrics, the three-judge rating protocol, the agent loop |
| Not in this repository | the images and annotations |

Chest Imagenome derives from MIMIC-CXR and RadGenome-BrainMRI from the BraTS
cohorts. Both require credentialed access, so the pixels cannot be
redistributed here. What ships instead is everything needed to rebuild the
exact evaluation set once you hold the source data.

## Cases

| Task | Cases | Source | Anatomy / Modality | Dim. |
|---|---:|---|---|---|
| `organ_anomaly_grounding` | 55 | Chest Imagenome | Chest / X-ray | 2D |
| `diagnosis_with_clues` | 55 | RadGenome-BrainMRI | Head & Neck / MRI | 3D |
| `report_generation` | 55 | RadGenome-BrainMRI | Head & Neck / MRI | 3D |

The brain MRI cases are sampled across BraTS-GLI, BraTS-MEN and BraTS-MET
(glioma, meningioma, metastasis), balanced as far as availability allows.

Build the manifest from your local copies:

```bash
python RealEnv/build_manifest.py \
    --chest-imagenome /data/chest-imagenome \
    --radgenome /data/radgenome-brainmri \
    --out RealEnv/manifest.json
```

Sampling is seeded, so the same `--seed` over the same source data reproduces
the same manifest. `CaseManifest.validate()` refuses a manifest that is not
55/55/55, since a 54-case run reported as the benchmark is the drift this
guards against.

**Question inputs versus reference answers.** `RealCase.inputs` holds what the
clinical question already states (which organ to localise, say) and seeds the
agent's memory bank. `RealCase.reference` holds the answer. They are separate
fields on purpose: merging them would leak the label into the input.

## Tools

14 dataset-specific families × 3 subset-trained variants = **42 executable
tools**, plus nnU-Net and RP3D-DM as ready-made alternatives.

| Group | Families |
|---|---|
| Organ Segmentor | BTCV, VerSe, OASIS |
| Anomaly Detector | COVID19, Instance22, MSD_Pancreas, ChestX-Det10 |
| Disease Diagnoser | ADNI, KneeMRI, CC-CCII, CT-kidney |
| Report Generator | CT-RATE, BrainGnome, IU-Xray |

Three variants per family is what makes the perturbation settings possible on
real tools: Differentiated needs several genuinely suitable candidates that
differ in quality, Redundant needs tools that look alike in name while
differing in scope.

```python
from RealEnv import build_default_pool

pool = build_default_pool()                                   # 44 tools
pool.candidates("Anomaly Detector", "Chest", "X-ray", "2D")   # 3, best first
pool.without_category("Report Generator")                     # config1
pool.without_capability("Report Generator", "Chest", "X-ray") # config2
pool.best_only()                                              # baseline
```

The agent sees `tool.card()`, which **omits performance**. A benchmark where
the agent can read tool quality off an oracle field measures nothing; pass
`card(reveal_performance=True)` only when the condition is meant to reveal
it.

Performance values start at 0.0. Fill them from real validation runs with
`calibrate_from_results` before using the Differentiated setting, or every
tool ties and OTS becomes meaningless.

## Running an evaluation

The executable tools live outside this repository and differ per installation,
so the thing that actually invokes them is injected:

```bash
python RealEnv/run_real_eval.py \
    --model claude-sonnet-4.5 \
    --manifest RealEnv/manifest.json \
    --executor mytools.runner:run_tool \
    --task report_generation \
    --llm-rating \
    --out Outputs/real
```

`--executor` names a `module:function` receiving `(ExecutableTool, inputs)` and
returning a dict keyed like the tool card's `output`. There is no built-in
fallback: a stub executor returning plausible arrays would produce numbers
that look like results and are not.

## Metrics

| Task | Automatic | Also |
|---|---|---|
| Grounding | Dice, Jaccard | LLM Rating, Human Rating |
| Diagnosis | Accuracy, Jaccard (over label sets) | LLM Rating, Human Rating |
| Report | RadGraph-F1, RaTE Score | LLM Rating, Human Rating |

Dice, Jaccard and Accuracy are implemented here, since they are simple and a
library dependency would make results drift with a version bump. RadGraph-F1
and RaTE wrap their reference implementations (`pip install radgraph
RaTEScore`) and are **not** approximated: a lexical stand-in would produce
numbers that cannot be compared with any published figure.

**LLM Rating** averages over cases *and* three judges. Per-judge means and
inter-judge disagreement are kept in the output, so a split panel is visible
rather than averaged away. **Human Rating** merges in from a reviewer file via
`--human-ratings`.

**The idealized-executable gap.** `reported_ceiling_gap(U, C)` computes
Δ = (U − C)/C × 100%. U substitutes a tool's officially reported performance
into the idealized protocol; C is the column-wise best across systems. Neither
is one deployable agent's score, so this is a computational reference point,
not headroom.

## AutoTB scenarios

660 scenarios, 220 per tier, instantiated by hiding something each case needs:

| Tier | Condition | What is hidden |
|---|---|---|
| Regular | config1 | a required tool category |
| Intermediate | config2 | that category's coverage of this anatomy/modality |
| Complex | config3 | the capability, with the category left visible |

config3 is hardest precisely because the category is still there: an agent
that checks only "is there a Report Generator?" answers yes and proceeds.

```python
from RealEnv.autotb_scenarios import build_scenarios, run_all_scenarios

scenarios = build_scenarios(manifest, pool)          # 660
report = run_all_scenarios(scenarios, agent_factory, runner, registry)
```

`agent_factory` is a factory rather than a fixed agent because the replan must
see the *augmented* pool. Reusing the original agent would replan against the
same insufficient tools, and the build would have achieved nothing.

## Files

| File | Role |
|---|---|
| `cases.py` | manifest schema, loading, validation |
| `build_manifest.py` | rebuild the 165-case set from source datasets |
| `tool_pool.py` | the 42-variant executable pool and its perturbations |
| `agent.py` | the executable agent loop |
| `metrics.py` | Dice, Jaccard, Accuracy, RadGraph-F1, RaTE, the gap |
| `llm_rating.py` | three-judge clinical rating |
| `autotb_scenarios.py` | 660 scenarios and the closed AutoTB loop |
| `run_real_eval.py` | CLI |
