"""Prompting strategies for the agent core.

    from Prompting import compose

    strategy = compose("fewshot,multiagent")
    system = strategy.system_prompt(initial_prompt)
"""

from .strategies import (STRATEGIES, BackPropagation, FewShot, MultiAgent,
                         SelfReflection, Strategy, compose, parse_strategy_list)

__all__ = ["Strategy", "BackPropagation", "SelfReflection", "FewShot",
           "MultiAgent", "STRATEGIES", "compose", "parse_strategy_list"]
