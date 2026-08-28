# AutoTB: automated tool building

When the tool set cannot solve a task, a capable agent should not hallucinate
a tool. It should stop, say precisely what is missing, and ask for that thing
to be built. AutoTB evaluates whether agents do this, and in the real
environment builds what they ask for.

The two benchmark layers share a protocol up to the point where a **tool card**
exists; they differ only in what happens next.

```
                     plan ──► executable? ──yes──► run
                                  │no
                                  ▼
                   ground the missing capability
                                  │
                    ┌─────────────┴─────────────┐
              not grounded                  grounded
                    │                           │
                 refuse                  specify tool card
                                                │
                    ┌───────────────────────────┴──────────────┐
            synthetic layer                              real layer
                    │                                          │
        mark card as available                   M3Builder × K trials
        (no training, no weights)                          │
                    │                              validate ─┬─ fail ─► construction failure
                    │                                        │
                    │                                    register
                    └──────────────► replan ◄───────────────┘
```

## Synthetic layer

No model is trained. A case succeeds only when the agent both triggers AutoTB
and grounds the field its condition removed:

| Condition | The agent must name |
|---|---|
| `config1`, a whole category is absent | the missing **category** |
| `config2`, modality/anatomy mismatch | **anatomy and modality** matching the case (`Universal` counts) |
| `config3`, required capability absent | the affected **category** *plus* the `InsufficientCapability` marker |

`config3` is the hardest and, per the paper, the main bottleneck: an agent can
recognise the tool set is inadequate while still asking for the wrong thing.
That is why *awareness* and *grounding* are counted separately here rather than
collapsed into one number.

Expected request format:

```xml
<NoCall>
  <Purpose>Generate a report for the chest X-ray</Purpose>
  <Category>Report Generator</Category>
  <Anatomy>Chest</Anatomy>
  <Modality>X-ray</Modality>
  <Ability>SpecificToolMissing</Ability>
</NoCall>
```

Score a run:

```bash
python AutoTB/run_synthetic.py \
    --logs Outputs/claude-sonnet-4.5/multiDenyl2/logouts \
    --condition config2 \
    --out Outputs/autotb_config2.json
```

## Real-environment layer

Here the request is executed. [M³Builder](https://github.com/MAGIC-AI4Med/M3Builder)
trains a candidate, the candidate is validated against the tool card, and only
a validated candidate is registered in the executable pool.

```bash
git clone https://github.com/MAGIC-AI4Med/M3Builder.git
```

```python
from AutoTB import run_real_autotb
from AutoTB.m3builder import M3BuilderRunner, ToolPool

runner = M3BuilderRunner("/path/to/M3Builder", max_trials=3)
pool = ToolPool("RealEnv/tool_pool.json")

result = run_real_autotb(
    case_id="chest-0007",
    transcript=agent_log,
    condition="config2",
    case=case,
    runner=runner,
    pool=pool,
    replan_and_run=lambda tool_id: agent.replan_with(tool_id),
)
```

M³Builder's four agents (Dataset Selector, Data Processor, Dataloader Builder,
Model Trainer) run in sequence, and the wrapper records which stage was
reached. That is what makes the failure attribution below possible.

## Failure attribution

The categories are separated because they call for different fixes.

| Mode | What happened | Where the fix lies |
|---|---|---|
| `PLANNING_ERROR` | No build request was ever emitted | prompting |
| `UNGROUNDED_REQUEST` | A request was made, naming the wrong field | grounding / reasoning |
| `CONSTRUCTION_FAILURE` | M³Builder produced no validated tool | data, templates, environment |
| `USAGE_ERROR` | Tool built and registered, then called wrongly | I/O alignment |
| `SUCCESS` | Built (or simulated), registered, used correctly | n/a |

The boundary is the validated candidate. Anything failing before one exists
(dataset selection, a non-executable dataloader, training termination, a
missing checkpoint, absent evaluation output) is construction. Anything
failing after registration, because a call supplied an incompatible input,
omitted a compulsory one, or requested an unsupported anatomy/modality, is
usage. Collapsing the two would make the failure-mode analysis meaningless.

## Metrics

`summarize()` returns AutoTB Trigger Rate, Grounding Rate, Execution Success
Rate, and Average Build Time, the quantities reported in the paper's stability
and failure-mode tables.

## Files

| File | Role |
|---|---|
| `verifier.py` | Parse a build request; check grounding per condition |
| `tool_card.py` | Turn a verified request into a tool card; map to an M³Builder task |
| `m3builder.py` | Invoke M³Builder, validate the result, register into the pool |
| `protocol.py` | Algorithm 1 for both layers; failure attribution; aggregation |
| `run_synthetic.py` | CLI for scoring synthetic-layer transcripts |
