# Synthetic-layer construction pipeline

How the 2,200 patient records and 24,200 QA pairs are built
(paper Appendix `sec:construction_pipeline`).

```
raw <Case> XML ──1──► structured records ──2──► QA text ──3──► evaluation file ──4──► xlsx
                 parse                  generate      combine                  export
```

Stage 2 is the one that costs money and time. Stages 1, 3 and 4 are pure data
transformations and are safe to re-run.

```bash
# everything at once, 5 records per anatomy-modality combination
python DataPipeline/build_dataset.py all \
    --raw Cases/ --workdir build/ --model gpt-4 --subset 5

# or one stage at a time
python DataPipeline/build_dataset.py parse-cases --raw Cases/ --out Processed/
python DataPipeline/build_dataset.py generate-qa --cases build/records.json --out QA/ --model gpt-4
python DataPipeline/build_dataset.py combine-qa  --cases build/records.json --qa QA/ --out Data/eval.json
python DataPipeline/build_dataset.py export-xlsx --data Data/eval.json --out Data/eval.xlsx
```

`generate-qa` skips records that already have output, so an interrupted run
resumes instead of paying for the same records twice.

## The generation prompt

`qa_generation_prompt.txt` defines the ten-tool toolkit and the eleven task
templates. It is kept as a file rather than inlined because it *is* the
dataset specification, and paraphrasing it produces a different dataset. It is
verbatim what generated the published QA pairs.

## Subsetting

`--subset N` takes the first N records of each anatomy-modality block rather
than N × 22 at random, so every one of the 22 combinations stays represented.
`--subset 5` gives the 110-record subset shipped in `Data/`.

## Verified against the published data

Re-parsing the raw generation output reproduces
`Data/combined_casesV3_qa_Subset.json` **byte for byte across all 110 cases**.

One thing that surfaced while checking: six cases (`400`, `402`, `601`,
`1902`, `1903`, `2004`) carry 10 task types rather than 11, so the subset holds
1,204 QA pairs rather than 1,210. `combine-qa` keeps them by default, because
dropping them would fail to reproduce the published file, and reports the
count. Pass `--require-complete` to drop them instead, which is worth doing if
an even per-task sample size matters for what you are measuring.

## Records, before QA

Records are generated from structured templates covering 22 anatomy-modality
combinations × 100, then polished for over a month by a radiologist with 10+
years of experience (97.32% information validity, 96.73% clinical
consistency), with three further clinicians reviewing 550 sampled cases across
eight criteria (average PPA 0.880-0.935).

`parse-cases` handles the output of that generation step; the generation
prompts for the records themselves are part of the release on
[HuggingFace](https://huggingface.co/datasets/QiaoyuZheng/RadABench-Data).
