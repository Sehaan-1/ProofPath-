"""Model-backed wiring shared by analysis start and approval resume.

`build_workflow` needs a role parser, embedding client, comparison backend and
sprint planner. Starting a run and resuming an approval gate must therefore build
them the same way; this module is that single construction point.

Construction is lazy and cheap: no network call happens until a node uses a
dependency. Resuming an approval gate never re-enters a model-backed node, but the
graph is still compiled with the real wiring so a resumed thread can never run
with silently disabled dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from proofpath.config import Settings
from proofpath.graph.backends import create_graph_backend
from proofpath.graph.nodes import EvidenceComparator, SprintPlanner
from proofpath.ingestion.analyse_role import RoleParserBackend, create_role_parser
from proofpath.retrieval.embeddings import EmbeddingClient, create_embedding_client


class GraphDependenciesError(RuntimeError):
    """The configured provider cannot back a graph run or resume."""


@dataclass(frozen=True, slots=True)
class GraphDependencies:
    """Everything `build_workflow` needs besides persistence and settings."""

    parser: RoleParserBackend
    embedder: EmbeddingClient
    comparator: EvidenceComparator
    sprint_planner: SprintPlanner


def create_graph_dependencies(settings: Settings) -> GraphDependencies:
    """Build the configured adapters or fail with one sanitized error."""
    try:
        backend = create_graph_backend(settings)
        parser = create_role_parser(settings)
        embedder = create_embedding_client(settings)
    except ValueError as exc:
        raise GraphDependenciesError("Analysis model is not configured.") from exc
    return GraphDependencies(
        parser=parser,
        embedder=embedder,
        comparator=backend,
        sprint_planner=backend,
    )
