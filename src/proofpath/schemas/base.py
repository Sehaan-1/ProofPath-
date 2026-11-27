"""Shared Pydantic v2 base for ProofPath request, response, and LLM contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Reject unexpected keys so LLM-hallucinated arguments fail fast.

    Sub-part 1.2.1b: `extra="forbid"` is the default for every ProofPath contract,
    including tool-input schemas used by MCP / LangGraph tool calls.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        from_attributes=True,
        use_enum_values=False,
    )
