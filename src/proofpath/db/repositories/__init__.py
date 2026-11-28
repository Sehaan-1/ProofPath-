"""Typed async SQLAlchemy repositories for ProofPath domain tables."""

from proofpath.db.repositories.analysis import AnalysisRepo, ClaimVerdictInput
from proofpath.db.repositories.audit import AuditRepo
from proofpath.db.repositories.calendar_task import CalendarTaskRepo
from proofpath.db.repositories.draft import DraftRepo
from proofpath.db.repositories.evidence import EvidenceChunkInput, EvidenceRepo
from proofpath.db.repositories.opportunity import OpportunityRepo, RoleRequirementInput
from proofpath.db.repositories.tracker import TrackerRepo

__all__ = [
    "AnalysisRepo",
    "AuditRepo",
    "CalendarTaskRepo",
    "ClaimVerdictInput",
    "DraftRepo",
    "EvidenceChunkInput",
    "EvidenceRepo",
    "OpportunityRepo",
    "RoleRequirementInput",
    "TrackerRepo",
]
