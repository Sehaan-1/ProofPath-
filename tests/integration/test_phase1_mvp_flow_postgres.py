"""End-to-end Phase 1 Exit Criteria verification against live PostgreSQL + pgvector.

Phase 1 Exit Criteria (`proofpath_build_plan.md`):
1. Ingest a resume + 2 GitHub repos via FastAPI endpoints.
2. Submit a realistic Palo Alto Networks (PANW) job description via API.
3. Run the bounded LangGraph workflow and receive a cited `supported / partial /
   unsupported` report (including deterministic policy downgrade of hallucinated
   support for `SAP S/4HANA`), a 2-5 day contribution sprint, and
   `pending_approval` action drafts without mutating `application_tracker`.
4. Query the exact same PostgreSQL ledger, opportunity, tracker status, and
   pending action drafts over the local ProofPath MCP server.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
import pytest
import respx
from mcp import Client
from pydantic import PostgresDsn
from sqlalchemy import delete, select

from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    AnalysisRunStatus,
    AuditLog,
    EvidenceSource,
    Opportunity,
    RequirementCategory,
    RequirementPriority,
    VerdictType,
)
from proofpath.graph.state import PolicyViolation
from proofpath.main import create_app
from proofpath.mcp_server import create_mcp_server
from proofpath.policy.ledger_validator import CLAIM_DOWNGRADED_EVENT
from proofpath.schemas import (
    ActionDraftRecord,
    AnalysisReport,
    ApplicationStatusResponse,
    ClaimVerdictSchema,
    ContributionSprintPlan,
    EvidenceSearchResponse,
    EvidenceSnippet,
    OpportunityResponse,
    ParsedRole,
    RequirementItem,
    RequirementRecord,
)

pytestmark = pytest.mark.integration

_PANW_JD_TEXT = """
Palo Alto Networks — Senior AI Agent & Platform Engineer (Cortex AI)

About the Role:
Join Palo Alto Networks to build trustworthy, citation-grounded AI agent workflows
and developer tooling for enterprise security operations.

Must-Have Qualifications:
- Strong production experience in Python and asynchronous backend services with FastAPI.
- Hands-on experience building hybrid RAG retrieval pipelines on PostgreSQL with pgvector.
- Experience designing and shipping Model Context Protocol (MCP) servers and agent workflows.
- Enterprise ERP administration and customization in SAP S/4HANA finance modules.
"""

_RESUME_MARKDOWN = """# Alex Rivera — AI Backend Engineer

## Experience
Built asynchronous Python and FastAPI microservices handling high-throughput API traffic,
typed Pydantic v2 schemas, and PostgreSQL persistence with SQLAlchemy 2.0 and Alembic.

## Projects
Designed hybrid semantic and lexical search pipelines on PostgreSQL 16 using pgvector HNSW
cosine indexing combined with ts_rank Reciprocal Rank Fusion.

## Skills
Python, FastAPI, PostgreSQL, pgvector, SQLAlchemy, Alembic, Docker, PyTest, Ruff, Mypy.

## Education
B.S. in Computer Science.
"""


def _mock_github_repo(
    router: respx.MockRouter,
    *,
    owner: str,
    name: str,
    description: str,
    languages: dict[str, int],
    readme: str,
    pr_title: str,
    pr_body: str,
) -> None:
    base = f"https://api.github.com/repos/{owner}/{name}"
    router.get(base).mock(
        return_value=httpx.Response(
            200,
            json={
                "full_name": f"{owner}/{name}",
                "html_url": f"https://github.com/{owner}/{name}",
                "description": description,
                "default_branch": "main",
                "stargazers_count": 42,
            },
        )
    )
    router.get(f"{base}/languages").mock(return_value=httpx.Response(200, json=languages))
    router.get(f"{base}/readme").mock(return_value=httpx.Response(200, text=readme))
    router.get(f"{base}/releases").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "tag_name": "v0.2.0",
                    "name": "v0.2.0",
                    "body": f"Initial public release of {name}.",
                    "html_url": f"https://github.com/{owner}/{name}/releases/tag/v0.2.0",
                }
            ],
        )
    )
    router.get(f"{base}/pulls").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "number": 7,
                    "title": pr_title,
                    "body": pr_body,
                    "html_url": f"https://github.com/{owner}/{name}/pull/7",
                }
            ],
        )
    )


async def test_phase1_exit_criteria_end_to_end_api_langgraph_and_mcp(
    integration_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=PostgresDsn(integration_database_url),
        embedding_model="hash",
        max_agent_retries=1,
    )
    app = create_app(settings=settings)

    run_suffix = uuid.uuid4().hex[:8]
    resume_uri = f"file:///resumes/alex_rivera_{run_suffix}.md"
    repo_one_owner = "alexrivera"
    repo_one_name = f"rag-pgvector-{run_suffix}"
    repo_two_owner = "alexrivera"
    repo_two_name = f"mcp-prototype-{run_suffix}"

    class DeterministicPANWRoleParser:
        async def parse(self, parser_prompt: str) -> ParsedRole:
            assert "<untrusted_job_description>" in parser_prompt
            assert "Palo Alto Networks" in parser_prompt
            return ParsedRole(
                company="Palo Alto Networks",
                role_title="Senior AI Agent & Platform Engineer",
                summary="Build citation-grounded AI agent workflows and MCP services.",
                requirements=[
                    RequirementItem(
                        requirement_text="Production experience in Python and FastAPI backend services.",
                        entity="FastAPI",
                        category=RequirementCategory.FRAMEWORK,
                        priority=RequirementPriority.MUST_HAVE,
                    ),
                    RequirementItem(
                        requirement_text="Hybrid RAG retrieval pipelines on PostgreSQL with pgvector.",
                        entity="pgvector",
                        category=RequirementCategory.FRAMEWORK,
                        priority=RequirementPriority.MUST_HAVE,
                    ),
                    RequirementItem(
                        requirement_text="Designing and shipping Model Context Protocol (MCP) servers.",
                        entity="MCP",
                        category=RequirementCategory.AI_AGENT,
                        priority=RequirementPriority.MUST_HAVE,
                    ),
                    RequirementItem(
                        requirement_text="Enterprise ERP administration in SAP S/4HANA finance modules.",
                        entity="SAP S/4HANA",
                        category=RequirementCategory.DOMAIN,
                        priority=RequirementPriority.MUST_HAVE,
                    ),
                ],
            )

    class DeterministicGraphBackend:
        """Comparator that cites real retrieved chunks for grounded skills and
        intentionally attempts an ungrounded `supported` claim on `SAP S/4HANA`
        so `policy_verifier_node` proves deterministic unsupported-claim rejection.
        """

        async def compare(
            self,
            requirement: RequirementRecord,
            evidence: list[EvidenceSnippet],
            feedback: list[PolicyViolation],
        ) -> ClaimVerdictSchema:
            matching = [
                chunk
                for chunk in evidence
                if requirement.entity.casefold() in chunk.content.casefold()
            ]
            if requirement.entity in {"FastAPI", "pgvector"}:
                assert matching, f"Expected hybrid retrieval hit for {requirement.entity}"
                return ClaimVerdictSchema(
                    requirement_id=requirement.id,
                    verdict=VerdictType.SUPPORTED,
                    claim_summary=f"Verified {requirement.entity} ownership across resume and GitHub.",
                    cited_chunk_ids=[chunk.chunk_id for chunk in matching[:2]],
                )
            if requirement.entity == "MCP":
                assert matching, "Expected GitHub prototype hit for MCP"
                return ClaimVerdictSchema(
                    requirement_id=requirement.id,
                    verdict=VerdictType.PARTIAL,
                    claim_summary="Prototype MCP stdio experiment exists in GitHub repository.",
                    cited_chunk_ids=[matching[0].chunk_id],
                    gap_notes="Needs streamable HTTP transport and Postgres integration tests.",
                )
            # `SAP S/4HANA` is absent from the ledger: simulate an over-eager LLM claiming
            # `supported` with an unrelated chunk so `LedgerValidator` forces `unsupported`.
            unrelated_citation = [evidence[0].chunk_id] if evidence else []
            return ClaimVerdictSchema(
                requirement_id=requirement.id,
                verdict=VerdictType.SUPPORTED,
                claim_summary="Hallucinated SAP S/4HANA support.",
                cited_chunk_ids=unrelated_citation,
            )

        async def plan(
            self,
            requirement: RequirementRecord,
            verdict: ClaimVerdictSchema,
        ) -> ContributionSprintPlan:
            return ContributionSprintPlan(
                title=f"Close {requirement.entity} gap with reproducible OSS demo",
                scope=(
                    f"Build a focused, tested reference integration addressing "
                    f"{requirement.entity} ({verdict.verdict.value})."
                ),
                repo_or_topic=f"github.com/{repo_two_owner}/{repo_two_name}",
                deliverables=[
                    "End-to-end integration test suite",
                    "Architecture README with reproducible benchmarks",
                ],
                estimate_days=3,
                target_gap_requirement_ids=[requirement.id],
                rationale="Must-have unsupported requirement has highest hiring signal impact.",
            )

    monkeypatch.setattr(
        "proofpath.api.routes.create_role_parser",
        lambda _settings: DeterministicPANWRoleParser(),
    )
    monkeypatch.setattr(
        "proofpath.api.analysis_routes.create_role_parser",
        lambda _settings: DeterministicPANWRoleParser(),
    )
    monkeypatch.setattr(
        "proofpath.api.analysis_routes.create_graph_backend",
        lambda _settings: DeterministicGraphBackend(),
    )

    created_source_ids: list[uuid.UUID] = []
    requirement_ids: list[uuid.UUID] = []
    opportunity_id: uuid.UUID | None = None
    analysis_run_id: uuid.UUID | None = None
    thread_id: str | None = None

    async with app.router.lifespan_context(app):
        try:
            with respx.mock(assert_all_called=True) as github_mock:
                _mock_github_repo(
                    github_mock,
                    owner=repo_one_owner,
                    name=repo_one_name,
                    description="Production FastAPI + pgvector hybrid RAG retrieval service",
                    languages={"Python": 18500},
                    readme=(
                        f"# {repo_one_name}\n\n"
                        "## Overview\n"
                        "FastAPI service backed by PostgreSQL 16 and pgvector hybrid RRF search.\n"
                    ),
                    pr_title="Add pgvector HNSW cosine index and ts_rank fusion",
                    pr_body="Improves exact acronym recall in FastAPI evidence search.",
                )
                _mock_github_repo(
                    github_mock,
                    owner=repo_two_owner,
                    name=repo_two_name,
                    description="Experimental Model Context Protocol (MCP) tool server",
                    languages={"Python": 9200},
                    readme=(
                        f"# {repo_two_name}\n\n"
                        "## Prototype\n"
                        "Early MCP read-only server prototype exploring tool JSON schemas.\n"
                    ),
                    pr_title="Initial MCP stdio prototype",
                    pr_body="Adds experimental MCP tool definitions in Python.",
                )

                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test",
                ) as api_client:
                    health = await api_client.get("/health")
                    assert health.status_code == 200
                    assert health.json() == {
                        "status": "ok",
                        "postgres": "ok",
                        "pgvector": "ok",
                    }

                    # 1) Ingest real multi-section Resume + 2 GitHub repos via API.
                    resume_resp = await api_client.post(
                        "/api/v1/evidence/ingest/file",
                        files={
                            "file": (
                                "alex_rivera_resume.md",
                                _RESUME_MARKDOWN.encode("utf-8"),
                                "text/markdown",
                            )
                        },
                        data={"canonical_uri": resume_uri, "source_type": "resume"},
                    )
                    assert resume_resp.status_code == 201, resume_resp.text
                    resume_body = resume_resp.json()
                    assert resume_body["created"] is True
                    assert resume_body["chunk_count"] >= 3
                    created_source_ids.append(uuid.UUID(resume_body["source_id"]))

                    gh_one_resp = await api_client.post(
                        "/api/v1/evidence/ingest/github",
                        json={"repo_url": f"https://github.com/{repo_one_owner}/{repo_one_name}"},
                    )
                    assert gh_one_resp.status_code == 201, gh_one_resp.text
                    gh_one_body = gh_one_resp.json()
                    assert gh_one_body["source_type"] == "github_repo"
                    assert (
                        gh_one_body["canonical_uri"]
                        == f"https://github.com/{repo_one_owner}/{repo_one_name}"
                    )
                    created_source_ids.append(uuid.UUID(gh_one_body["source_id"]))

                    gh_two_resp = await api_client.post(
                        "/api/v1/evidence/ingest/github",
                        json={"owner": repo_two_owner, "name": repo_two_name},
                    )
                    assert gh_two_resp.status_code == 201, gh_two_resp.text
                    gh_two_body = gh_two_resp.json()
                    assert gh_two_body["source_type"] == "github_repo"
                    assert (
                        gh_two_body["canonical_uri"]
                        == f"https://github.com/{repo_two_owner}/{repo_two_name}"
                    )
                    created_source_ids.append(uuid.UUID(gh_two_body["source_id"]))

                    # 2) Submit PANW job description via API.
                    opp_resp = await api_client.post(
                        "/api/v1/opportunities",
                        json={
                            "raw_text": _PANW_JD_TEXT,
                            "source_url": "https://jobs.paloaltonetworks.com/en/job/ai-agent-engineer",
                        },
                    )
                    assert opp_resp.status_code == 201, opp_resp.text
                    opp_model = OpportunityResponse.model_validate(opp_resp.json())
                    opportunity_id = opp_model.id
                    requirement_ids = [r.id for r in opp_model.requirements]
                    assert opp_model.company == "Palo Alto Networks"
                    assert {r.entity for r in opp_model.requirements} == {
                        "FastAPI",
                        "pgvector",
                        "MCP",
                        "SAP S/4HANA",
                    }

                    # 3) Execute bounded LangGraph analysis workflow via API.
                    run_resp = await api_client.post(
                        "/api/v1/analyses/run",
                        json={
                            "opportunity_id": str(opportunity_id),
                            "calendar_start": datetime(2026, 12, 1, 9, 0, tzinfo=UTC).isoformat(),
                            "calendar_end": datetime(2026, 12, 4, 17, 0, tzinfo=UTC).isoformat(),
                        },
                    )
                    assert run_resp.status_code == 201, run_resp.text
                    report = AnalysisReport.model_validate(run_resp.json())
                    analysis_run_id = report.analysis_run_id
                    thread_id = report.thread_id

                    assert report.status is AnalysisRunStatus.AWAITING_APPROVAL
                    assert report.errors == []

                    req_by_entity = {r.entity: r for r in report.requirements}
                    verdict_by_req = {v.requirement_id: v for v in report.verdicts}

                    # Supported skills carry real ledger chunk citations.
                    for supported_entity in ("FastAPI", "pgvector"):
                        v = verdict_by_req[req_by_entity[supported_entity].id]
                        assert v.verdict is VerdictType.SUPPORTED
                        assert len(v.cited_chunk_ids) >= 1
                        assert v.policy_downgraded is False

                    # Partial skill carries citation + explicit gap notes.
                    mcp_verdict = verdict_by_req[req_by_entity["MCP"].id]
                    assert mcp_verdict.verdict is VerdictType.PARTIAL
                    assert len(mcp_verdict.cited_chunk_ids) == 1
                    assert mcp_verdict.gap_notes is not None

                    # Unsupported skill (`SAP S/4HANA`) was deterministically downgraded by policy.
                    sap_verdict = verdict_by_req[req_by_entity["SAP S/4HANA"].id]
                    assert sap_verdict.verdict is VerdictType.UNSUPPORTED
                    assert sap_verdict.policy_downgraded is True

                    # Contribution sprint targets the must-have unsupported gap (`SAP S/4HANA`).
                    assert report.contribution_sprint is not None
                    assert report.contribution_sprint.target_gap_requirement_ids == [
                        req_by_entity["SAP S/4HANA"].id
                    ]
                    assert 2 <= report.contribution_sprint.estimate_days <= 5

                    # Pending-approval action drafts created; zero tracker writes executed.
                    assert {d.action_type for d in report.action_drafts} == {
                        ActionType.TRACKER_UPSERT,
                        ActionType.CALENDAR_BLOCK,
                    }
                    assert all(
                        d.status is ActionDraftStatus.PENDING_APPROVAL for d in report.action_drafts
                    )

                    tracker_resp = await api_client.get("/api/v1/tracker")
                    assert tracker_resp.status_code == 200
                    assert not any(
                        entry["opportunity_id"] == str(opportunity_id)
                        for entry in tracker_resp.json()
                    )

                    fetched_report_resp = await api_client.get(
                        f"/api/v1/analyses/{analysis_run_id}"
                    )
                    assert fetched_report_resp.status_code == 200
                    assert AnalysisReport.model_validate(fetched_report_resp.json()) == report

            # Confirm deterministic policy audit log was persisted in Postgres.
            sap_req_id = req_by_entity["SAP S/4HANA"].id
            async with app.state.session_factory() as session:
                audit_rows = list(
                    (
                        await session.execute(
                            select(AuditLog).where(
                                AuditLog.resource_id == sap_req_id,
                                AuditLog.event_type == CLAIM_DOWNGRADED_EVENT,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                assert len(audit_rows) >= 1
                assert all(
                    row.details_json.get("analysis_run_id") == str(analysis_run_id)
                    for row in audit_rows
                )

            # 4) Query the EXACT SAME PostgreSQL ledger, opportunity, tracker & drafts over MCP.
            mcp_server = create_mcp_server(settings=settings)
            async with Client(mcp_server) as mcp_client:
                listed = await mcp_client.list_tools()
                assert {t.name for t in listed.tools} == {
                    "search_evidence",
                    "get_opportunity",
                    "get_application_status",
                    "get_action_draft",
                }

                # Search evidence over MCP and verify chunks from ingested resume + GitHub repos.
                mcp_search = await mcp_client.call_tool(
                    "search_evidence",
                    {"query": "FastAPI pgvector Reciprocal Rank Fusion", "top_k": 8},
                )
                assert mcp_search.is_error is False
                search_payload = EvidenceSearchResponse.model_validate(
                    mcp_search.structured_content
                )
                returned_source_ids = {item.source_id for item in search_payload.results}
                assert returned_source_ids & set(created_source_ids)
                returned_chunk_ids = {item.chunk_id for item in search_payload.results}
                cited_supported_ids = set(
                    verdict_by_req[req_by_entity["pgvector"].id].cited_chunk_ids
                )
                assert returned_chunk_ids & cited_supported_ids

                # Read opportunity over MCP.
                mcp_opp = await mcp_client.call_tool(
                    "get_opportunity",
                    {"opportunity_id": str(opportunity_id)},
                )
                assert mcp_opp.is_error is False
                mcp_opp_model = OpportunityResponse.model_validate(mcp_opp.structured_content)
                assert mcp_opp_model.id == opportunity_id
                assert mcp_opp_model.company == "Palo Alto Networks"
                assert len(mcp_opp_model.requirements) == 4

                # Check application tracker status over MCP (still untracked pending approval).
                mcp_status = await mcp_client.call_tool(
                    "get_application_status",
                    {"opportunity_id": str(opportunity_id)},
                )
                assert mcp_status.is_error is False
                status_model = ApplicationStatusResponse.model_validate(
                    mcp_status.structured_content
                )
                assert status_model.opportunity_id == opportunity_id
                assert status_model.tracked is False
                assert status_model.stage is None

                # Read each `pending_approval` action draft over MCP without executing it.
                for draft_summary in report.action_drafts:
                    mcp_draft = await mcp_client.call_tool(
                        "get_action_draft",
                        {"draft_id": str(draft_summary.id)},
                    )
                    assert mcp_draft.is_error is False
                    draft_model = ActionDraftRecord.model_validate(mcp_draft.structured_content)
                    assert draft_model.id == draft_summary.id
                    assert draft_model.status is ActionDraftStatus.PENDING_APPROVAL
                    assert draft_model.approved_at is None
                    assert draft_model.executed_at is None
        finally:
            if thread_id is not None:
                await app.state.checkpointer.adelete_thread(thread_id)
            audit_cleanup_ids = list(requirement_ids)
            if analysis_run_id is not None:
                audit_cleanup_ids.append(analysis_run_id)
            async with app.state.session_factory.begin() as cleanup_session:
                if audit_cleanup_ids:
                    await cleanup_session.execute(
                        delete(AuditLog).where(AuditLog.resource_id.in_(audit_cleanup_ids))
                    )
                if opportunity_id is not None:
                    await cleanup_session.execute(
                        delete(Opportunity).where(Opportunity.id == opportunity_id)
                    )
                if created_source_ids:
                    await cleanup_session.execute(
                        delete(EvidenceSource).where(EvidenceSource.id.in_(created_source_ids))
                    )
