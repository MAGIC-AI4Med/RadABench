"""Automated tool building.

Synthetic layer: verify that an agent facing an insufficient tool set asks for
the right thing. Real layer: build it with M3Builder, validate it, use it.

    from AutoTB import run_synthetic_autotb, summarize

    results = [run_synthetic_autotb(cid, text, "config2", case)
               for cid, text, case in cases]
    print(summarize(results))
"""

from .protocol import (FAILURE_MODES, AutoTBResult, run_real_autotb,
                       run_synthetic_autotb, summarize)
from .tool_card import CATEGORY_IO, ToolCard, specify_tool_card
from .verifier import BuildRequest, VerificationResult, parse_request, verify

__all__ = [
    "AutoTBResult", "FAILURE_MODES", "run_synthetic_autotb", "run_real_autotb",
    "summarize", "ToolCard", "specify_tool_card", "CATEGORY_IO",
    "BuildRequest", "VerificationResult", "parse_request", "verify",
]
