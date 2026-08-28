# Simulated tool sets

Each file is a tool pool that the agent core is given for one evaluation
condition. The filenames predate the paper and do **not** match its
terminology: `toy_tools.txt` is the Baseline condition, while `basic_tools.txt`
is already a redundant set. Use this table, not the names.

| File | Unique tools | Paper condition (§2.1, Fig. 1d) | Tests |
|---|---:|---|---|
| `toy_tools.txt` | 12 | **Baseline**, minimal solvable set, 10 categories, BQ and IC have two tools each | Can the agent orchestrate at all? |
| `basic_tools.txt` | 15 | **Redundant**, regular, at most one extra tool per category | Filtering under light noise |
| `scale_tools.txt` | 34 | **Redundant**, medium, 2-3 extras per category | Dependency tracking among similar candidates |
| `whole_tools.txt` | 169 | **Redundant**, high, fixed 169-tool pool | Selection under heavy distraction |
| `deny1_tools.txt` | 17 | **Insufficient**, config1, an entire category is absent | Recognising a missing *category* |
| `deny2_tools.txt` | 17 | **Insufficient**, config2, modality/anatomy mismatch | Recognising a missing *capability for this case* |
| `deny2Complex_tools.txt` | 17 | **Insufficient**, config2 restricted to complex tasks | Same, on long chains |
| `deny3_tools.txt` | 18 | **Insufficient**, config3, required capability missing | Hardest unsolvability grounding |
| `performance_tools.txt` | 18 | **Differentiated**, overlapping tools at distinct performance levels | Picking the *most accurate* tool |

The three Insufficient configs are ordered by recognition difficulty:
config1 (a whole category is gone) is the most visible, config3 (the category
is present but no tool has the needed capability) the least.

## Format

Every entry is a **tool card**: category, properties, capabilities, required
and optional inputs, and a performance figure sampled from a clinically
plausible range. `Utils/tooluse.py` parses these into `ToolManager_IO`, which
exposes them to the agent as `TOOLINFO` / `TOOLDES` / `TOOLBOX`.

## Which pipeline loads which set

| Script | Tool set |
|---|---|
| `EvalPipeline_multihop_general.py`, `EvalPipeline_singlehop_general.py` | Baseline |
| `EvalPipeline_multihop_IO.py` | Redundant regular |
| `EvalPipeline_multihop_IO_large.py` | Redundant medium |
| `EvalPipeline_multihop_IO_scale.py` | Redundant high |
| `EvalPipeline_multihop_denyl1/2/3.py` | Insufficient config1/2/3 |
| `EvalPipeline_multihop_performance.py` | Differentiated |
