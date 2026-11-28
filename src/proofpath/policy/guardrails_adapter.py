"""Opt-in Guardrails AI schema validator. No Hub install, LLM call, repair or re-ask."""

from __future__ import annotations

from importlib import import_module

from proofpath.policy.output_guard import OutputFacts, OutputGuardUnavailableError
from proofpath.policy.tool_allowlist import ANALYSIS_TOOL_NAMES


class GuardrailsAIOutputGuard:
    def validate(self, facts: OutputFacts) -> None:
        # Load only on explicit opt-in. Use the SDK's local payload validator rather
        # than its generation/server API; this adapter needs no prompts or API keys.
        try:
            validate_payload = import_module("guardrails.schema.validator").validate_payload
        except (ImportError, AttributeError) as exc:
            raise OutputGuardUnavailableError(
                "ENABLE_GUARDRAILS requires the proofpath[guardrails] extra."
            ) from exc

        payload = {
            "grounded_citations": [
                [str(cid) for cid in group] for group in facts.grounded_citations
            ],
            "tool_trace": list(facts.tool_trace),
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["grounded_citations", "tool_trace"],
            "properties": {
                "grounded_citations": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"enum": sorted(str(cid) for cid in facts.existing_chunk_ids)}
                        if facts.existing_chunk_ids
                        else False,
                    },
                },
                "tool_trace": {"type": "array", "items": {"enum": sorted(ANALYSIS_TOOL_NAMES)}},
            },
        }
        validate_payload(payload, schema)
