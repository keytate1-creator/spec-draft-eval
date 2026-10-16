"""
SpecDraft: Speculative Decoding Rejection-Sampling & Verification Engine.
"""

from spec_engine.sampler import RejectionSampler, VerificationDecision
from spec_engine.tree_draft import DraftTreeNode, DraftTree, TreeVerificationResult
from spec_engine.analyzer import SpeculativeSpeedupAnalyzer, SpeedupMetrics

__version__ = "0.1.0"
__author__ = "Kaylor Tate Cespedes"

__all__ = [
    "RejectionSampler",
    "VerificationDecision",
    "DraftTreeNode",
    "DraftTree",
    "TreeVerificationResult",
    "SpeculativeSpeedupAnalyzer",
    "SpeedupMetrics",
]
