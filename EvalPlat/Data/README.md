# Data

## What ships in the repository

`combined_casesV3_qa_Subset.json`: **110 patient records, 1,204 QA pairs**.
This is the evaluation subset used for the per-condition experiments; it is
small enough to version here so that results can be reproduced without any
download step.

Each record carries demographics, medical history, annotated imaging findings
(anatomy and pathology), and clinical findings, plus the QA pairs generated
from the 11 task templates. Task types are keyed by their tool-chain
signature, e.g. `"012"` = anatomy → modality → organ segmentation.

## The full release

The complete synthetic layer, **2,200 records (22 anatomy-modality
combinations x 100) and 24,200 QA pairs (2,200 x 11)**, is published separately
because of its size:

<https://huggingface.co/datasets/QiaoyuZheng/RadABench-Data>

Point any pipeline at it with `--data`:

```bash
python EvalPipeline_multihop_general.py \
    --model claude-sonnet-4.5 \
    --data /path/to/RadABench-Data/combined_cases_full.json
```

## Which data backs which result

| Paper result | Data |
|---|---|
| Table 1, completion rate by difficulty | full release |
| Table 2, tool-chain metrics under redundancy | full release |
| Figure 4, differentiated / insufficient / redundant analysis | full release |
| Per-condition ablations and smoke tests | the subset shipped here |
| Table 4, real-case evaluation | `RealEnv/` (165 real cases, not synthetic) |

## Provenance

Records were generated with GPT-4 from structured templates, then polished for
over a month by a radiologist with 10+ years of experience: 97.32% met
information-validity standards and 96.73% were judged clinically consistent.
Three further clinicians independently reviewed 550 sampled cases across eight
criteria (average PPA 0.880-0.935; paper Table 1).

The generation pipeline itself is in `DataPipeline/`.
