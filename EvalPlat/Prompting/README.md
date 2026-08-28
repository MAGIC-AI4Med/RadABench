# Prompting strategies

Four strategies that target the coordination bottlenecks rather than the base
model. They compose, and each hooks only the stage it actually changes, which
keeps them independently ablatable.

| Strategy | Flag | What it changes | Paper finding |
|---|---|---|---|
| Prompt Back-propagation | `backprop` | System prompt, refined over three rounds of feedback on 220 QA pairs | ~3% downstream correction; unstable alone |
| Self-reflection | `reflection` | Verifies each call's inputs before execution | Unstable alone |
| Few-shot Learning | `fewshot` | Exemplar workflows and tool templates | Significant (*p* < 0.05) |
| Multi-agent Collaboration | `multiagent` | Splits the core into planner / executor / summarizer | Most consistent; +~5% unsolvability awareness and grounding |

Combined, they lift task completion by 0.095 and 0.058 on simple and moderate
tasks, and by up to **48.2%** on complex ones.

```bash
python EvalPipeline_multihop_general.py --model claude-sonnet-4.5              # baseline
python EvalPipeline_multihop_general.py --strategies fewshot,multiagent
python EvalPipeline_multihop_general.py --strategies all                       # integrated
```

Outputs land under `Outputs/<model>/<run tag>+<strategies>/`, so an ablation
does not overwrite its baseline.

## Notes on each

**Back-propagation** ships the *final prompt*, not the procedure that produced
it. The procedure cannot be re-run without the same 220 QA pairs and manual
passes, so publishing the text is what makes the reported gain reproducible.

**Self-reflection** costs one extra model call per step. It catches invented
variables and out-of-scope tools before they become runtime errors, but it can
also talk the agent out of a correct call, which is why the paper finds it
unstable alone. A "correction" that comes back as prose rather than a call
block is rejected and the original kept.

**Few-shot** carries three exemplars. The third, refusing when no tool fits,
matters as much as the two that succeed, because agents otherwise rarely refuse
at all.

**Multi-agent** helps exactly where the single-core agent is weakest: long
chains. Separating planning from execution stops a mid-chain error from
rewriting the plan; separating summarization stops the executor's last tool
output from being mistaken for the answer.

## Writing another one

Subclass `Strategy` and override whichever of the three hooks applies:

```python
class MyStrategy(Strategy):
    name = "mine"

    def system_prompt(self, base): ...       # before the run
    def before_step(self, messages, state): ...  # before each model call
    def after_step(self, output, state, chat): ...  # before parsing
```

Register it in `STRATEGIES` and it becomes available to `--strategies`.
